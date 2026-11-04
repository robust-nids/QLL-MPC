#!/usr/bin/env bash
# Build the EzPC compiler and the SCI library (SecFloat + Beacon) inside the container.
# Run from the host:
#   docker run --rm --platform linux/amd64 -v "$PWD":/work beacon-sci bash docker/build.sh
set -euo pipefail

EZPC=/work/EzPC

# 1. EzPC compiler -> EzPC/EzPC/EzPC/ezpc (what compile_networks.py calls)
make -C "$EZPC/EzPC/EzPC"

# 2. SCI -> EzPC/SCI/build/install (as in SCI/README.md)
cmake -S "$EZPC/SCI" -B "$EZPC/SCI/build" -DCMAKE_INSTALL_PREFIX="$EZPC/SCI/build/install"
cmake --build "$EZPC/SCI/build" --target install --parallel 8
