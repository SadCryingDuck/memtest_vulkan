#!/bin/bash
(
    set -xeu
    # the same checks as the lint job in .github/workflows/ci.yml
    cargo fmt --all --check
    cargo clippy --all --all-targets --locked -- -D warnings
    cargo test --all --locked
)
