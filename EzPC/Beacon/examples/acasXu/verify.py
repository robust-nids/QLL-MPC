# run with:
#   python verify.py [networks] [models] [properties] [--verbose] [--timeout seconds]
#   [networks] = official, official finetuned, finetuned
#   [models] = 1.1,1.2,...,5.9, all
#   [properties] = 0,1,...,9
#   --verbose also prints the counterexamples found
#   --timeout is how long Marabou gets per query (default 30 s)

# run vehicle on the above models and properties, logs saved in logs/
# results.csv columns -> network type, model number, property number tested, result(error, falsified, verified, timeout,), seconds
#   then the timeout, and the finetune.py settings stored in a finetuned network (empty for official ones)

# print a progress bar
import argparse
import csv
import os
import signal
import subprocess
import time

import onnx
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = {"official": "official_nets", "finetuned": "finetuned_nets"}
SETTINGS = ["sampling", "samples", "box_size", "epochs", "batch", "lr", "p", "margin", "data", "data_weight"]

def find_networks(network_type, models):
    """the .onnx files of network_type for models ("1.1,1.2" or "all"); a model's finetuned
    networks are every acasXu_<x>_<y>_*.onnx"""
    folder = os.path.join(HERE, FOLDERS[network_type])
    names = sorted(name for name in os.listdir(folder) if name.endswith(".onnx"))
    if models != "all":
        prefixes = tuple(f"acasXu_{model.replace('.', '_')}" for model in models.split(","))
        names = [name for name in names if name.startswith(prefixes)]
    return [os.path.join(folder, name) for name in names]

def verify(path, prop, timeout=30):
    """one `vehicle verify` of property prop on the network at path, Marabou getting `timeout` seconds per query;
    returns (result, seconds, counterexample)"""
    name = os.path.basename(path).removesuffix(".onnx")
    cmd = ["vehicle", "verify",
           "--specification", os.path.join(HERE, "acasXu.vcl"),
           "--network", f"acasXu:{path}",
           "--property", f"property{prop}",
           "--solver", "Marabou",
           "--solver-args", f"--timeout={timeout}"]
    start = time.time()
    # own process group, so a timeout also kills the Marabou processes Vehicle started
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out = proc.communicate(timeout=5 * timeout)[0]  # Marabou sometimes ignores --timeout
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out = proc.communicate()[0] + f"\nverify.py: killed after {5 * timeout} s, Marabou timed out\n"
    seconds = time.time() - start
    with open(os.path.join(HERE, "logs", f"{name}_property{prop}.log"), "w") as f:
        f.write(" ".join(cmd) + "\n\n" + out)

    if "proved no counterexample exists" in out:
        return "verified", seconds, None
    if "found a counterexample" in out:
        return "falsified", seconds, out.split("found a counterexample", 1)[1].strip()  # vehicle prints it last
    if "timed out" in out:
        return "timeout", seconds, None
    return "error", seconds, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("networks", nargs="+", choices=FOLDERS)
    parser.add_argument("models", help="1.1,1.2,...,5.9 or all")
    parser.add_argument("properties", help="0,1,...,9")
    parser.add_argument("--verbose", action="store_true", help="also print the counterexamples found")
    parser.add_argument("--timeout", type=int, default=30, help="seconds Marabou gets per query")
    args = parser.parse_args()

    runs = [(network_type, path, int(prop))
            for network_type in args.networks
            for path in find_networks(network_type, args.models)
            for prop in args.properties.split(",")]

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    results = os.path.join(HERE, "results.csv")
    new_file = not os.path.exists(results)
    with open(results, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["network type", "model", "property", "result", "seconds", "timeout"]
                            + [setting.replace("_", " ") for setting in SETTINGS])
        for network_type, path, prop in tqdm(runs):
            result, seconds, counterexample = verify(path, prop, args.timeout)
            model = os.path.basename(path).removesuffix(".onnx").removeprefix("acasXu_")
            settings = {p.key: p.value for p in onnx.load(path).metadata_props}  # how it was finetuned
            writer.writerow([network_type, model, prop, result, f"{seconds:.1f}", args.timeout]
                            + [settings.get(setting, "") for setting in SETTINGS])
            f.flush()
            tqdm.write(f"{network_type:<10} {model:<10} property{prop}  {result:<9} {seconds:6.1f} s")  # above the bar
            if args.verbose and counterexample:
                tqdm.write(f"    {counterexample}")

if __name__ == "__main__":
    main()
