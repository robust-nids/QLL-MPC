## Running the experiment

### Requirements

- Docker. The image is `linux/amd64`: it runs natively on x86_64 and under emulation on Apple Silicon.
- Python 3.11 with the packages in [`ezpc/EzPC/Beacon/requirements.txt`](./ezpc/EzPC/Beacon/requirements.txt), only for the cleartext reference:
  `python3.11 -m pip install -r ezpc/EzPC/Beacon/requirements.txt`

### Setup

Run from `ezpc/`, which the container mounts as `/work`:

```bash
cd ezpc

# 1. Fetch the SCI build files and pinned dependencies
bash docker/fetch_upstream.sh

# 2. Build the image
docker build --platform linux/amd64 -t beacon-sci docker/

# 3. Build the EzPC compiler and SCI inside the container
docker run --rm --platform linux/amd64 -v "$PWD":/work beacon-sci bash docker/build.sh
```

Re-run step 3 after changing files in `EzPC/SCI/src`, and after building `ezpc` on the host:
host and container share the one `EzPC/EzPC/EzPC/ezpc` file, and its interpreter line is whichever build ran last.

### Training

From `ezpc/`:

```
docker run --rm --platform linux/amd64 -v "$PWD":/work -e NETWORK=ToyNetwork -e LOSS=TorchQLL -e QLL_P=2.0 -e BATCH=128 beacon-sci bash docker/train.sh
```

This trains ToyNetwork with the `TorchQLL` loss and p = 2. Both parties (server and client) run in the container. The script prints the revealed loss for each iteration, followed by the time, communication (MiB) and rounds.

| Variable                       | Values                                                                                    | Default        |
| ------------------------------ | ----------------------------------------------------------------------------------------- | -------------- |
| `NETWORK` | `ToyNetwork`, `Logistic`, `FFNN` (MNIST) | Logistic |
| `LOSS`    | `CE`, `MSE`, and for `ToyNetwork` only: `QLL` (hand-written C++ loss), `TorchQLL` (loss traced from PyTorch) | CE |
| `QLL_P`                      | QLL softness p                                                                            | 2.0            |
| `BATCH` / `ITERS` / `LR` | batch size, iterations, learning rate                                                     | 128 / 5 / 0.01 |

With `TorchQLL`, the property is the `loss()` function in [`beacon_frontend.py`](./EzPC/Beacon/beacon_frontend.py). It must match the chosen network's inputs and outputs.

### Cleartext reference for ToyNetwork

The same loss can be trained in plain PyTorch, on the same initial weights and data, to compare against the secure run. Run it from `EzPC/Beacon` after a training run, with the same loss, batch, iterations, lr and p.

```bash
python3 qll_reference.py ToyNetwork 128 5 0.01 2.0   # network, batch, iterations, lr, qll_p
```
