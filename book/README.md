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

Проверить встроенные Rust-фрагменты:

```bash
mdbook test book
```

Готовая HTML-версия будет находиться в `book/book/`.
