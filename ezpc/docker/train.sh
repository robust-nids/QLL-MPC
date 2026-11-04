#   docker run --rm --platform linux/amd64 -v "$PWD":/work beacon-sci bash docker/train.sh
# Override any setting with -e, e.g.  -e NETWORK=ToyNetwork -e ITERS=10 -e BACKEND=secfloat
set -euo pipefail

NETWORK=${NETWORK:-Logistic}   # Logistic, FFNN, Relevance, ToyNetwork
BATCH=${BATCH:-128}            # batch size
ITERS=${ITERS:-5}              # training iterations (the same batch is reused each time)
LR=${LR:-0.01}                 # learning rate
LOSS=${LOSS:-CE}               # CE (MSE writes a <network>_target file instead of labels)
MOMENTUM=${MOMENTUM:-no}       # yes or no
BACKEND=${BACKEND:-beacon}     # beacon or secfloat: which of the two built binaries to run
QLL_P=${QLL_P:-2.0}

BINARY=./${NETWORK}_${BACKEND}
WEIGHTS=${NETWORK}_weights.inp
INPUTS=${NETWORK}_input${BATCH}.inp
LABELS=${NETWORK}_labels${BATCH}.inp

cd /work/EzPC/Beacon

# compile_networks.py ignores build failures, so never fall back to stale binaries
rm -f "./${NETWORK}_secfloat" "./${NETWORK}_beacon" "./${NETWORK}.ezpc" "./${NETWORK}.cpp" "./${NETWORK}0.cpp"

echo "== Step 1: generate the training program and build it"
echo "   python3 compile_networks.py $NETWORK $BATCH $ITERS $LR $LOSS $MOMENTUM $QLL_P"
python3 compile_networks.py "$NETWORK" "$BATCH" "$ITERS" "$LR" "$LOSS" "$MOMENTUM" "$QLL_P"
[ -x "$BINARY" ] || { echo "Build failed: $BINARY was not produced (see the output above)"; exit 1; }

echo
echo "== Step 2: start the SERVER, which holds the weights (background, output in server.log)"
echo "   $BINARY r=1 < $WEIGHTS"
"$BINARY" r=1 < "$WEIGHTS" > server.log 2>&1 &
SERVER_PID=$!

echo
echo "== Step 3: run the CLIENT, which holds the data, and connect to the server"
echo "   cat $INPUTS $LABELS | $BINARY r=2 add=127.0.0.1"
cat "$INPUTS" "$LABELS" | "$BINARY" r=2 add=127.0.0.1 | tee client.log
wait "$SERVER_PID"

echo
echo "== Loss per iteration ($LOSS, $BACKEND), as revealed to both parties"
# The generated code prints loss[0] under its compiler temp name, "Value of __tac_var...".
# For CE it is the mean log-likelihood, i.e. -CE (Beacon's getLoss does not negate).
grep "Value of" client.log | awk '{printf "   iteration %d: %s\n", NR, $NF}'
