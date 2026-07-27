#!/usr/bin/env bash

# Универсальная perf-профилировка Cargo examples.
#
# По умолчанию для указанного примера собираются профили и статистика в двух
# режимах: с одним потоком Rayon и со стандартным параллелизмом. Профиль Cargo
# `profiling` наследует release-оптимизации, а этот скрипт дополнительно включает
# frame pointers для восстановления стеков.

set -Eeuo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROFILE_ROOT="${PROFILE_DIR:-$PROJECT_DIR/out/profiles}"
readonly PERF_EVENTS="${PERF_EVENTS:-task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,cache-references,cache-misses}"
readonly PERF_STAT_REPEATS="${PERF_STAT_REPEATS:-3}"
readonly PERF_RECORD_FREQUENCY="${PERF_RECORD_FREQUENCY:-499}"
readonly PARALLEL_THREADS="${PARALLEL_THREADS:-}"

usage() {
    cat <<'EOF'
Usage:
  ./profiling.sh [options] <example> [-- <example arguments...>]

Options:
  -p, --package <name>     Package containing the example (workspace root by default)
      --features <list>    Cargo feature list
      --all-features       Build with all Cargo features
      --serial-only        Profile only RAYON_NUM_THREADS=1
      --parallel-only      Profile only the default Rayon pool
      --build-only         Build the profiling binary without running perf
  -h, --help               Show this help

Environment:
  PROFILE_DIR              Output root (default: out/profiles)
  PERF_STAT_REPEATS        perf stat repeat count (default: 3)
  PERF_RECORD_FREQUENCY    perf record sampling frequency (default: 499)
  PERF_EVENTS              Comma-separated perf stat events
  PARALLEL_THREADS         Explicit Rayon size for parallel mode; unset uses its default

Examples:
  ./profiling.sh cbr_inflation_target
  ./profiling.sh --all-features faithful_mixture_fit
  ./profiling.sh -p gamlss-family --features multivariate objective_memory -- --help
EOF
}

require_option_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" ]]; then
        echo "error: $option requires a value" >&2
        usage >&2
        exit 2
    fi
}

example=""
package=""
features=""
all_features=false
serial=true
parallel=true
build_only=false
declare -a example_arguments=()

while (($# > 0)); do
    case "$1" in
        -p|--package)
            require_option_value "$1" "${2-}"
            package="$2"
            shift 2
            ;;
        --features)
            require_option_value "$1" "${2-}"
            features="$2"
            shift 2
            ;;
        --all-features)
            all_features=true
            shift
            ;;
        --serial-only)
            serial=true
            parallel=false
            shift
            ;;
        --parallel-only)
            serial=false
            parallel=true
            shift
            ;;
        --build-only)
            build_only=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            if [[ -z "$example" ]]; then
                echo "error: an example name is required before --" >&2
                usage >&2
                exit 2
            fi
            shift
            example_arguments=("$@")
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$example" ]]; then
                echo "error: example arguments must follow --" >&2
                usage >&2
                exit 2
            fi
            example="$1"
            shift
            ;;
    esac
done

if [[ -z "$example" ]]; then
    echo "error: an example name is required" >&2
    usage >&2
    exit 2
fi
if [[ ! "$example" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "error: invalid example name: $example" >&2
    exit 2
fi
if [[ -n "$package" && ! "$package" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "error: invalid package name: $package" >&2
    exit 2
fi
if [[ ! "$PERF_STAT_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: PERF_STAT_REPEATS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$PERF_RECORD_FREQUENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: PERF_RECORD_FREQUENCY must be a positive integer" >&2
    exit 2
fi

target_dir="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}"
if [[ "$target_dir" != /* ]]; then
    target_dir="$PROJECT_DIR/$target_dir"
fi
readonly TARGET_DIR="$target_dir"
readonly BINARY="$TARGET_DIR/profiling/examples/$example"

profile_name="$example"
if [[ -n "$package" ]]; then
    profile_name="$package-$example"
fi
readonly PROFILE_OUTPUT_DIR="$PROFILE_ROOT/$profile_name"

declare -a cargo_command=(cargo build --profile profiling --example "$example")
if [[ -n "$package" ]]; then
    cargo_command+=(--package "$package")
fi
if [[ -n "$features" ]]; then
    cargo_command+=(--features "$features")
fi
if [[ "$all_features" == true ]]; then
    cargo_command+=(--all-features)
fi

profiling_rustflags="-C force-frame-pointers=yes"
if [[ -n "${RUSTFLAGS-}" ]]; then
    profiling_rustflags="$RUSTFLAGS $profiling_rustflags"
fi

cd -- "$PROJECT_DIR"
mkdir -p -- "$PROFILE_OUTPUT_DIR"

echo "Building optimized profiling binary: $example"
env "RUSTFLAGS=$profiling_rustflags" "${cargo_command[@]}"
if [[ ! -x "$BINARY" ]]; then
    echo "error: Cargo did not produce executable $BINARY" >&2
    exit 1
fi
if [[ "$build_only" == true ]]; then
    echo "Built: $BINARY"
    exit 0
fi
if ! command -v perf >/dev/null 2>&1; then
    echo "error: perf is not installed or is not in PATH" >&2
    exit 1
fi

declare -a model_command=("$BINARY" "${example_arguments[@]}")

run_with_rayon_mode() {
    local mode="$1"
    shift

    case "$mode" in
        serial)
            env RAYON_NUM_THREADS=1 "$@"
            ;;
        parallel)
            if [[ -n "$PARALLEL_THREADS" ]]; then
                env "RAYON_NUM_THREADS=$PARALLEL_THREADS" "$@"
            else
                env -u RAYON_NUM_THREADS "$@"
            fi
            ;;
        *)
            echo "error: unsupported Rayon mode: $mode" >&2
            return 2
            ;;
    esac
}

rotate_previous_output() {
    local output="$1"
    if [[ -e "$output" ]]; then
        mv -f -- "$output" "$output.old"
    fi
}

record_profile() {
    local mode="$1"
    local output="$2"
    rotate_previous_output "$output"

    run_with_rayon_mode "$mode" perf record \
        --freq "$PERF_RECORD_FREQUENCY" \
        --event cycles:u \
        --call-graph fp \
        --output "$output" \
        -- "${model_command[@]}"
}

collect_stats() {
    local mode="$1"
    local output="$2"
    rotate_previous_output "$output"

    run_with_rayon_mode "$mode" perf stat \
        --repeat "$PERF_STAT_REPEATS" \
        --event "$PERF_EVENTS" \
        --output "$output" \
        -- "${model_command[@]}"
}

declare -a modes=()
if [[ "$serial" == true ]]; then
    modes+=(serial)
fi
if [[ "$parallel" == true ]]; then
    modes+=(parallel)
fi

for mode in "${modes[@]}"; do
    echo "Recording $mode call stacks"
    record_profile "$mode" "$PROFILE_OUTPUT_DIR/$mode.perf.data"
done
for mode in "${modes[@]}"; do
    echo "Collecting $mode counters ($PERF_STAT_REPEATS repeats)"
    collect_stats "$mode" "$PROFILE_OUTPUT_DIR/$mode.perf-stat.txt"
done

echo "Profiles written to: $PROFILE_OUTPUT_DIR"
report_mode="${modes[0]}"
if [[ "$parallel" == true ]]; then
    report_mode="parallel"
fi
echo "Inspect a call profile with: perf report --input $PROFILE_OUTPUT_DIR/$report_mode.perf.data"
