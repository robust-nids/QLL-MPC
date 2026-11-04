#!/usr/bin/env bash
# Restore the build files and pinned dependencies this repo is missing, from upstream.
# Run on the host from the repo root:  bash docker/fetch_upstream.sh
set -euo pipefail

EZPC_SHA=f24bf3e022dde3493cf0ffe3ebd92dd73936b734   # mpc-msri/EzPC master; SCI/Beacon sources here are identical to it
SEAL_SHA=1d5c8169aa5aca9deb75c4079e53ea8d5e94007d   # SEAL v3.3.2, upstream's SCI/extern/SEAL submodule pin
EIGEN_SHA=603e213d13311af286c8c1abd4ea14a8bd3d204e  # upstream's SCI/extern/eigen submodule pin

cd "$(dirname "$0")/../EzPC"

# 1. SCI CMake build files (dropped from this repo)
for f in SCI/CMakeLists.txt SCI/src/CMakeLists.txt \
         SCI/src/{utils,OT,GC,Millionaire,BuildingBlocks,LinearOT,LinearHE,NonLinear,Math,FloatingPoint}/CMakeLists.txt; do
  curl -fsSL "https://raw.githubusercontent.com/mpc-msri/EzPC/$EZPC_SHA/$f" -o "$f"
done

# 2. Submodules that SCI's CMake would otherwise `git submodule update` (impossible without .git)
if [ ! -f SCI/extern/SEAL/native/src/CMakeLists.txt ]; then
  mkdir -p SCI/extern/SEAL
  curl -fsSL "https://github.com/microsoft/SEAL/archive/$SEAL_SHA.tar.gz" | tar -xz --strip-components=1 -C SCI/extern/SEAL
  # SCI's CMake applies this right after cloning SEAL
  patch -d SCI/extern/SEAL -p1 < SCI/cmake/seal.patch
fi

if [ ! -f SCI/extern/eigen/CMakeLists.txt ]; then
  mkdir -p SCI/extern/eigen
  curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_SHA/eigen-$EIGEN_SHA.tar.gz" | tar -xz --strip-components=1 -C SCI/extern/eigen
fi
