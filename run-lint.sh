#!/bin/bash
(
    set -xeu
    # the same checks as the lint job in .github/workflows/ci.yml
    cargo fmt --all --check
    cargo clippy --all --all-targets --locked -- -D warnings
    cargo test --all --locked
    # CI lints on Linux: on Windows, also lint the Linux build, which compiles different code
    # (needs `rustup target add x86_64-unknown-linux-gnu`)
    if [[ "${OS:-}" == "Windows_NT" ]]; then
        cargo clippy --all --all-targets --locked --target x86_64-unknown-linux-gnu -- -D warnings
    fi
)
