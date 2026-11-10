#!/bin/bash
# Fine-tune ACAS Xu on property 1 in MPC (programs/acasXu.hpp, FUNCTION_IDENTIFIER=67) with all parties on this machine:
#   scripts/acasXu.sh mac
# Regenerates programs/functions/generated/acasXu_generated.hpp from torch/qll/nn.py and loss.py first.
# ACASXU_DIR holds P1's weights.inp and property.inp and P2's data.inp; P1 writes finetuned.inp there.
# Settings can be overridden from the environment, e.g. ITERS=390 LEARNING_RATE=1e-3 scripts/acasXu.sh mac
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-mac}
BATCH_SIZE=${BATCH_SIZE:-256}
ITERS=${ITERS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATA_POINTS=${DATA_POINTS:-9984}
PROPERTY_POINTS=${PROPERTY_POINTS:-43200}
REPORT_EVERY=${REPORT_EVERY:-10}
FRACTIONAL=${FRACTIONAL:-20}
export ACASXU_DIR=${ACASXU_DIR:-$PWD/examples/acasXu_run}

python3 programs/torch/torch_to_hpmpc.py --batch "$BATCH_SIZE"

# ripple-carry comparisons (the 4-way adder gave wrong signs) and TRUNC_APPROACH=1 (fixed_point_training.hpp truncates GEMM's products)
FLAGS="PROTOCOL=5 PRE=0 FUNCTION_IDENTIFIER=67 BITLENGTH=64 DATTYPE=64 FRACTIONAL=$FRACTIONAL TRUNC_APPROACH=1
       BANDWIDTH_OPTIMIZED=1 ONLINE_OPTIMIZED=0 BATCH_SIZE=$BATCH_SIZE ITERS=$ITERS LEARNING_RATE=$LEARNING_RATE
       DATA_POINTS=$DATA_POINTS PROPERTY_POINTS=$PROPERTY_POINTS REPORT_EVERY=$REPORT_EVERY"
if [ "$MODE" != "mac" ]; then
    echo "usage: $0 mac" >&2
    exit 1
fi
OPENSSL=${OPENSSL:-/opt/homebrew/opt/openssl@3}
BIN=executables-mac
mkdir -p "$BIN"
DEFS=""
for f in $FLAGS; do DEFS="$DEFS -D$f"; done
for p in 0 1 2; do
    clang++ main.cpp -std=c++20 -w -march=native -O2 -pthread -I nn/PIGEON \
        -I"$OPENSSL/include" -L"$OPENSSL/lib" -lssl -lcrypto -include core/include/pch.h \
        $DEFS -DPARTY=$p -DSPLIT_ROLES_OFFSET=0 -DUSE_SSL_AES=1 -DARM=0 -o "$BIN/acasXu-P$p.o" &
done
wait

for p in 0 1 2; do
    ./"$BIN"/acasXu-P$p.o > "$BIN/acasXu-P$p.log" 2>&1 &
done
wait
grep -h "ACASXU" "$BIN/acasXu-P1.log"
grep -h "computation chrono" "$BIN/acasXu-P1.log"