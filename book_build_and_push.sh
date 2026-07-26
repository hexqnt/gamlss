#!/usr/bin/env bash
set -euo pipefail

BOOK_DIR="${BOOK_DIR:-book}"
OUTPUT_DIR="${OUTPUT_DIR:-${BOOK_DIR}/book}"
S3_BUCKET="${S3_BUCKET:-gamlss.hexq.ru}"
S3_PREFIX="${S3_PREFIX:-}"
GZIP_MIN_BYTES="${GZIP_MIN_BYTES:-10240}"

S3_PREFIX="${S3_PREFIX#/}"
S3_PREFIX="${S3_PREFIX%/}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not available in PATH." >&2
    exit 1
  fi
}

content_type_for() {
  local file_path="$1"

  case "$file_path" in
    *.html) echo "text/html; charset=utf-8" ;;
    *.css) echo "text/css; charset=utf-8" ;;
    *.js | *.mjs) echo "text/javascript; charset=utf-8" ;;
    *.json | *.map) echo "application/json" ;;
    *.svg) echo "image/svg+xml" ;;
    *.xml) echo "application/xml" ;;
    *.wasm) echo "application/wasm" ;;
    *.png) echo "image/png" ;;
    *.jpg | *.jpeg) echo "image/jpeg" ;;
    *.gif) echo "image/gif" ;;
    *.webp) echo "image/webp" ;;
    *.avif) echo "image/avif" ;;
    *.ico) echo "image/x-icon" ;;
    *.woff) echo "font/woff" ;;
    *.woff2) echo "font/woff2" ;;
    *.ttf) echo "font/ttf" ;;
    *.eot) echo "application/vnd.ms-fontobject" ;;
    *.txt) echo "text/plain; charset=utf-8" ;;
    *.pdf) echo "application/pdf" ;;
    *) echo "application/octet-stream" ;;
  esac
}

is_compressible() {
  case "$1" in
    *.html | *.css | *.js | *.mjs | *.json | *.map | *.svg | *.xml | *.wasm | *.txt | *.ttf | *.eot) return 0 ;;
    *) return 1 ;;
  esac
}

target_base_uri() {
  if [[ -n "$S3_PREFIX" ]]; then
    printf 's3://%s/%s' "$S3_BUCKET" "$S3_PREFIX"
  else
    printf 's3://%s' "$S3_BUCKET"
  fi
}

cache_control_for() {
  case "$1" in
    *.html) echo "no-cache" ;;
    *) echo "public, max-age=3600" ;;
  esac
}

upload_file() {
  local src_path="$1"
  local rel_path="$2"
  local upload_path="$src_path"
  local object_key="$rel_path"
  local content_type
  local cache_control
  local source_size
  local compressed_size
  local encoding="identity"
  local -a upload_args

  if [[ -n "$S3_PREFIX" ]]; then
    object_key="${S3_PREFIX}/${rel_path}"
  fi

  content_type="$(content_type_for "$rel_path")"
  cache_control="$(cache_control_for "$rel_path")"
  source_size="$(wc -c < "$src_path")"
  upload_args=(
    --content-type "$content_type"
    --cache-control "$cache_control"
    --only-show-errors
  )

  if is_compressible "$rel_path" && ((source_size >= GZIP_MIN_BYTES)); then
    gzip -9 -n -c "$src_path" > "${TEMP_DIR}/upload.gz"
    upload_path="${TEMP_DIR}/upload.gz"
    compressed_size="$(wc -c < "$upload_path")"
    upload_args+=(--content-encoding gzip)
    encoding="gzip (${source_size} -> ${compressed_size} bytes)"
  fi

  yc storage s3 cp "$upload_path" "s3://${S3_BUCKET}/${object_key}" "${upload_args[@]}"
  echo "Uploaded: ${rel_path} [${encoding}]"
}

require_command find
require_command gzip
require_command mdbook
require_command mktemp
require_command rm
require_command sort
require_command wc
require_command yc

if [[ -z "$S3_BUCKET" ]]; then
  echo "Error: S3_BUCKET must not be empty." >&2
  exit 1
fi

if [[ ! "$GZIP_MIN_BYTES" =~ ^[0-9]+$ ]]; then
  echo "Error: GZIP_MIN_BYTES must be a non-negative integer." >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
readonly TEMP_DIR
trap 'rm -rf -- "$TEMP_DIR"' EXIT

echo "Building mdBook from ${BOOK_DIR}..."
mdbook build "$BOOK_DIR"

if [[ ! -f "${OUTPUT_DIR}/index.html" ]]; then
  echo "Error: ${OUTPUT_DIR}/index.html is missing; refusing to clean the bucket." >&2
  exit 1
fi

mapfile -d '' book_files < <(find "$OUTPUT_DIR" -type f -print0 | sort -z)
if [[ "${#book_files[@]}" -eq 0 ]]; then
  echo "Error: no files found in ${OUTPUT_DIR}; refusing to clean the bucket." >&2
  exit 1
fi

regular_files=()
html_files=()
for src_path in "${book_files[@]}"; do
  rel_path="${src_path#"$OUTPUT_DIR"/}"
  if [[ "$rel_path" == *.html ]]; then
    html_files+=("$src_path")
  else
    regular_files+=("$src_path")
  fi
done

TARGET_BASE_URI="$(target_base_uri)"
readonly TARGET_BASE_URI

echo "Cleaning remote objects at ${TARGET_BASE_URI}..."
yc storage s3 rm "$TARGET_BASE_URI" --recursive --only-show-errors

echo "Uploading static assets to ${TARGET_BASE_URI}..."
for src_path in "${regular_files[@]}"; do
  rel_path="${src_path#"$OUTPUT_DIR"/}"
  upload_file "$src_path" "$rel_path"
done

echo "Uploading HTML files to ${TARGET_BASE_URI}..."
for src_path in "${html_files[@]}"; do
  rel_path="${src_path#"$OUTPUT_DIR"/}"
  upload_file "$src_path" "$rel_path"
done

echo "Publish complete: ${TARGET_BASE_URI}"
