# Сборка книги

Книга использует препроцессор [`mdbook-katex`](https://github.com/lzanini/mdbook-katex), который преобразует TeX-формулы в HTML во время сборки.

Установите инструменты один раз:

```bash
cargo install mdbook
cargo install mdbook-katex --version 0.10.0-alpha
```

Собрать книгу:

```bash
mdbook build book
```

Проверить полные Rust-примеры, фрагменты которых включены в книгу:

```bash
cargo check --examples --all-features
```

Встроенные фрагменты не являются самостоятельными программами и поэтому помечены `rust,ignore`; команда `mdbook test book` не проверяет их компиляцию.

Готовая HTML-версия будет находиться в `book/book/`.
