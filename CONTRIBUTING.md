# Contributing to bobyqa-java

Thanks for your interest in contributing.

## Reporting bugs / proposing features

Open a [GitHub Issue](https://github.com/fair-acc/bobyqa-java/issues). For non-trivial design changes, please file the issue *before* a PR so we can align on the approach.

## Pull requests

1. Fork, branch off `main`, work in a feature branch.
2. Run the full build locally before pushing:
   ```bash
   mvn clean verify
   ```
   This runs the test suite and enforces the GPLv3 source-header check.
3. New `.java` files must carry the GPLv3 header. Run:
   ```bash
   mvn license:format
   ```
   to prepend headers automatically.
4. Follow the existing code style: Java 17, 4-space indent, no wildcard imports.
5. Use conventional-commit prefixes in commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `build:`, `chore:`.

## License

By contributing to this project you agree your contribution is licensed under **GPL-3.0-or-later**, the same license as the project.
