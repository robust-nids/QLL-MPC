# Clear text version of QLL as a reference

# python3 qll_reference.py ToyNetwork 128 5 0.01 2.0
# batch, iterations, learning rate, qll-p 
#%%
import argparse
import inspect
import math
import sys

import torch
import torch.nn.functional as F
from compile_networks import MNISTFFNN, MNISTLogistic, ToyNetwork
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import grad
#%% 
class Carrier:
    def __init__(self, p):
        self.p = p

class Mul(Carrier):
    def tensor(self, a, b): return a * b
    def par(self, a, b): return a * b
    def disj(self, a, b): return (a**self.p + b**self.p)**(1/self.p)
    def conj(self, a, b): return (a**-self.p + b**-self.p)**(-1/self.p)
    def implies(self, a, b): return self.par(self.dual(a), b)
    def dual(self, a): return 1/a

class Add(Carrier):
    def tensor(self, a, b): return a + b
    def par(self, a, b): return a + b
    def disj(self, a, b): return -(1/self.p) * torch.log(torch.exp(-a * self.p) + torch.exp(-b * self.p))
    def conj(self, a, b): return (1/self.p) * torch.log(torch.exp(a * self.p) + torch.exp(b * self.p))
    def implies(self, a, b): return self.par(self.dual(a), b)
    def dual(self, a): return -a

def to_add(a): return -torch.log(a)

def to_mul(a): return torch.exp(-a)

def loss(p, y0, x0, x1, label0):
    """Training loss for ONE sample, the only place the loss is stated.
    Parameter names say where each value comes from (as in beacon_frontend.py):
        yJ = network output J, xJ = network input J, labelJ = label J
    ToyNetwork: L = (y0 - label0)^2 + (|y0 - x0| \\/ |y0 - x1|)"""
    q = Mul(p)
    data = y0 - label0
    return data * data + q.disj((y0 - x0).abs(), (y0 - x1).abs())

def loss_args(y, x, label):
    """The batch column for each loss() parameter: y0 -> y[:, 0], x1 -> x[:, 1], label0 -> label[:, 0]"""
    arrays = {"y": y, "x": x, "label": label}
    names = list(inspect.signature(loss).parameters)[1:]
    return [arrays[n.rstrip("0123456789")][:, int(n[len(n.rstrip("0123456789")):])] for n in names]

def load_inp(path, dtype):
    with open(path) as f:
        return torch.tensor([float(v) for v in f.read().split()], dtype=dtype)
# %%
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("network")
    ap.add_argument("batch", type=int)
    ap.add_argument("iters", type=int)
    ap.add_argument("lr", type=float)
    ap.add_argument("qll_p", type=float, nargs="?", default=2.0)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float32")
    args = ap.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    net = ToyNetwork().to(dtype)
    torch.nn.utils.vector_to_parameters(load_inp(f"{args.network}_weights.inp", dtype), net.parameters())
    X = load_inp(f"{args.network}_input{args.batch}.inp", dtype).reshape(args.batch, -1)
    Y = load_inp(f"{args.network}_labels{args.batch}.inp", dtype).reshape(args.batch, -1)

    print(f"== Loss per iteration (reference {args.dtype}, p={args.qll_p})")
    for i in range(args.iters):
        y = net(X)
        L = loss(args.qll_p, *loss_args(y, X, Y)).mean()
        print(f"   iteration {i+1}: {L.item():.15g}")

        net.zero_grad()
        L.backward()
        with torch.no_grad():
            for prm in net.parameters():
                prm -= args.lr * prm.grad

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main(["ToyNetwork", "128", "5", "0.01", "2.0"])
    else:
        main()

# main(["ToyNetwork", "128", "5", "0.01", "2.0"])
# %%

# quick visualization 
toy_spec = lambda *args: loss(2.0, *args)

example = [torch.rand(()) for _ in list(inspect.signature(loss).parameters)[1:]]
loss_gm = make_fx(toy_spec)(*example)
der_gm = make_fx(grad(toy_spec))(*example)

print("==== LOSS GRAPH MODULE====")
for n in loss_gm.graph.nodes:
    print(n.op, n.name, n.target, n.args)
print("==== DERIVATIVE GRAPH MODULE ====") 
# note: derivatives are not mpc friendly
# write custom torch grad functions that are mpc friendly
for n in der_gm.graph.nodes:
    print(n.op, n.name, n.target, n.args)
# %%