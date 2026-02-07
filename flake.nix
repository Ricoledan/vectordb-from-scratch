{
  description = "vectordb-from-scratch - Building a vector database in Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rustToolchain

            # Build dependencies
            pkg-config
            openssl

            # Development tools
            cargo-watch      # Auto-rebuild on changes
            cargo-edit       # cargo add/rm/upgrade
            cargo-nextest    # Better test runner
            cargo-criterion  # Benchmarking

            # Optional: for BLAS acceleration
            # openblas
          ];

          shellHook = ''
            echo "🦀 Rust vector-db dev environment loaded"
            echo "Rust: $(rustc --version)"
            echo "Cargo: $(cargo --version)"
          '';

          # For rust-analyzer in VS Code
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };
      }
    );
}
