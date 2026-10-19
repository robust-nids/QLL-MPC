# run with:
#   python finetune.py [model] [properties] --sampling [uniform|region|boxes]
#   e.g. python finetune.py 1.1 2 3 4 --sampling region
#   python finetune.py 2.9 1 --sampling boxes --p 10000 --margin 0.002 --data-weight 10000 --lr 1e-4 --epochs 30
#     makes Vehicle prove property 1 on 2_9, with 97.1% / 98.1% accuracy on ART's orig / normed test sets

# imports
import argparse
import functools
import math
import os

import matplotlib.pyplot as plt
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "acasXu_data")

# load offical nets from onnx -> torch

def load_onnx(path):
    """an official or finetuned network as 5 -> 6 x 50 ReLU -> 5, from the MatMul/Add/Relu nodes of its
    graph. The official graphs first subtract input_AvgImg, which is zero in every official network."""
    graph = onnx.load(path).graph
    weights = {t.name: torch.tensor(numpy_helper.to_array(t)) for t in graph.initializer}
    layers = []
    for node in graph.node:
        w = next((weights[name] for name in node.input if name in weights), None)
        if node.op_type == "MatMul":
            layers.append(nn.Linear(*w.shape))
            with torch.no_grad():
                layers[-1].weight.copy_(w.T)  # ONNX MatMul is x @ W, nn.Linear is x @ weight.T
        elif node.op_type == "Add":
            with torch.no_grad():
                layers[-1].bias.copy_(w)
        elif node.op_type == "Relu":
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# define loss for 10 properties in QLL

# The conclusions of acasXu.vcl's properties as additive QLL (smaller is truer, -inf true, +inf false):
# a < b and a <= b are a - b, not is dual (-x), and is Add.conj, or is Add.disj.

class Interval:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

class Add:
    def __init__(self, p):
        self.p = p

    # logaddexp(x, y) = log(exp(x) + exp(y)), without overflowing for a large p.
    def disj(self, a, b):
        if isinstance(a, Interval):
            return Interval(self.disj(a.lo, b.lo), self.disj(a.hi, b.hi))
        return -torch.logaddexp(-a * self.p, -b * self.p) / self.p

    def conj(self, a, b):
        if isinstance(a, Interval):
            return Interval(self.conj(a.lo, b.lo), self.conj(a.hi, b.hi))
        return torch.logaddexp(a * self.p, b * self.p) / self.p

    def dual(self, a): return -a

COC_1500 = (1500 - 7.518884) / 373.94992   # property 0's threshold: scaleCOCOutput 1500

def qll_all(q, xs): return functools.reduce(q.conj, xs)   # x1 /\ x2 /\ ...

def minimal_score(q, i, y):
    """minimalScore i: y_i < y_j for every j != i"""
    return qll_all(q, [y[i] - y[j] for j in range(5) if j != i])

def maximal_score(q, i, y):
    """maximalScore i: y_i > y_j for every j != i"""
    return qll_all(q, [y[j] - y[i] for j in range(5) if j != i])

def prop0(p, y0, y1, y2, y3, y4):
    """COC score <= scaleCOCOutput 1500"""
    return y0 - COC_1500

def prop1(p, y0, y1, y2, y3, y4):
    """not (maximalScore clearOfConflict)"""
    q = Add(p)
    return q.dual(maximal_score(q, 0, (y0, y1, y2, y3, y4)))

def prop2(p, y0, y1, y2, y3, y4):
    """not (minimalScore clearOfConflict); property 3 has the same conclusion"""
    q = Add(p)
    return q.dual(minimal_score(q, 0, (y0, y1, y2, y3, y4)))

def prop4(p, y0, y1, y2, y3, y4):
    """minimalScore strongRight"""
    return minimal_score(Add(p), 4, (y0, y1, y2, y3, y4))

def prop5(p, y0, y1, y2, y3, y4):
    """minimalScore clearOfConflict; property 9 has the same conclusion"""
    return minimal_score(Add(p), 0, (y0, y1, y2, y3, y4))

def prop6(p, y0, y1, y2, y3, y4):
    """not (minimalScore strongLeft) and not (minimalScore strongRight)"""
    q, y = Add(p), (y0, y1, y2, y3, y4)
    return q.conj(q.dual(minimal_score(q, 3, y)), q.dual(minimal_score(q, 4, y)))

def prop7(p, y0, y1, y2, y3, y4):
    """minimalScore clearOfConflict or minimalScore weakLeft"""
    q, y = Add(p), (y0, y1, y2, y3, y4)
    return q.disj(minimal_score(q, 0, y), minimal_score(q, 1, y))

def prop8(p, y0, y1, y2, y3, y4):
    """minimalScore strongLeft"""
    return minimal_score(Add(p), 3, (y0, y1, y2, y3, y4))

PROPS = {0: prop0, 1: prop1, 2: prop2, 3: prop2, 4: prop4, 5: prop5, 6: prop6, 7: prop7, 8: prop8, 9: prop5}

def crown(net, lo, hi, A):
    """lower bounds of A @ net(x) over each box [lo, hi] (CROWN: every ReLU between two linear bounds,
    every layer bounded in turn); exact on a box of size 0"""
    linears = [layer for layer in net if isinstance(layer, nn.Linear)]
    centre, radius = (lo + hi) / 2, (hi - lo) / 2

    def back(A, const, n, pre):
        """lower bound of A @ (input of linears[n - 1]) + const, back through the hidden layers before it"""
        for linear, (l, u) in zip(reversed(linears[:n - 1]), reversed(pre)):
            unstable, active = (l < 0) & (u > 0), (l >= 0).float()
            up = torch.where(unstable, u / (u - l).clamp(min=1e-12), active)  # relu(z) <= up * (z - l)
            down = torch.where(unstable, (u > -l).float(), active)             # relu(z) >= down * z
            neg = A < 0
            const = const - (A * neg * (up * l * unstable)[:, None]).sum(-1)
            A = A * torch.where(neg, up[:, None], down[:, None])
            const, A = const + A @ linear.bias, A @ linear.weight
        return (A @ centre[..., None]).squeeze(-1) - (A.abs() @ radius[..., None]).squeeze(-1) + const

    pre = []  # bounds of every hidden pre-activation
    for n in range(1, len(linears)):
        W, b = linears[n - 1].weight, linears[n - 1].bias
        bound = back(torch.cat([W, -W]).expand(len(lo), -1, -1), torch.cat([b, -b]).expand(len(lo), -1), n, pre)
        pre.append((bound[:, :len(b)], -bound[:, len(b):]))
    last = linears[-1]
    return back((A @ last.weight).expand(len(lo), -1, -1), (A @ last.bias).expand(len(lo), -1), len(linears), pre)

class Score:
    """score i of the network over a batch of boxes; subtracting another Score or a number gives an Interval"""
    def __init__(self, i, bounds):
        self.i, self.bounds = i, bounds

    def __sub__(self, other):
        lo, hi, diff = self.bounds  # diff[:, i, j] is a lower bound of y_i - y_j
        if isinstance(other, Score):
            return Interval(diff[:, self.i, other.i], -diff[:, other.i, self.i])
        return Interval(lo[:, self.i] - other, hi[:, self.i] - other)

E = torch.eye(5)
BOUNDED = torch.cat([E, -E, (E[:, None] - E[None]).reshape(25, 5)])  # y_i, -y_i and y_i - y_j

def qll_bound(net, k, p, lo, hi):
    """an upper bound of property k's QLL value over each box [lo, hi]: below 0, it holds on the whole box"""
    b = crown(net, lo, hi, BOUNDED)
    bounds = (b[:, :5], -b[:, 5:10], b[:, 10:].reshape(-1, 5, 5))
    return PROPS[k](p, *[Score(i, bounds) for i in range(5)]).hi

# load data
#   either:
#       sammple data uniformly (hypothesis: we won't actually sample data relavant to QLL prop we are testing and will tank accuracy only)
#       sample data from input space relevant to QLL property that we are testing, i.e. why would we finetune a medical imagining CNN with photos of elephants, but idk so lets test it
#       could we do an adversairal type sampling?

def load_data(model, variant, split):
    """ART's dataset for a model: normalised inputs, labelled with the official network's advisory;
    orig is uniform over the inputs, normed has as many points of every advisory as ART could find"""
    x, y = model.split("_")
    X, labels = torch.load(os.path.join(DATA_DIR, f"AcasNetID<{x},{y}>-{variant}-{split}.pt"), weights_only=True)
    return X.float(), labels.long()

# acasXu.vcl's validInput ranges and normalisation: (x - mean) / (max - min)
PI = 3.141592
IN_MIN = torch.tensor([0.0, -PI, -PI, 100.0, 0.0])
IN_MAX = torch.tensor([60261.0, PI, PI, 1200.0, 1200.0])
IN_MEAN = torch.tensor([19791.091, 0.0, 0.0, 650.0, 600.0])

def normalise(x): return (x - IN_MEAN) / (IN_MAX - IN_MIN)

# uniform data sampling

def sample_uniform(n):
    """n uniform points in validInput, normalised like the network's inputs"""
    return normalise(IN_MIN + torch.rand(n, 5) * (IN_MAX - IN_MIN))

# input space relevant sampling

# each property's region in problem units, (low, high) for [rho, theta, psi, v_own, v_int];
# several boxes are a union (property 5's two theta ranges). Clipped to validInput when sampled.
REGIONS = {
    0: [[(55947.691, 60261), (-PI, PI), (-PI, PI), (1145, 1200), (0, 60)]],
    1: [[(55947.691, 60261), (-PI, PI), (-PI, PI), (1145, 1200), (0, 60)]],
    2: [[(1500, 1800), (-0.06, 0.06), (3.10, PI), (980, 1200), (960, 1200)]],
    3: [[(1500, 1800), (-0.06, 0.06), (0, 0), (1000, 1200), (700, 800)]],
    4: [[(250, 400), (0.2, 0.4), (-PI, -PI + 0.005), (100, 400), (0, 400)]],
    5: [[(12000, 62000), (-PI, -0.7), (-PI, -PI + 0.005), (100, 1200), (0, 1200)],
        [(12000, 62000), (0.7, PI), (-PI, -PI + 0.005), (100, 1200), (0, 1200)]],
    6: [[(0, 60760), (-PI, PI), (-PI, PI), (100, 1200), (0, 1200)]],
    7: [[(0, 60760), (-PI, -0.75 * PI), (-0.1, 0.1), (600, 1200), (600, 1200)]],
    8: [[(2000, 7000), (-0.4, -0.14), (-PI, -PI + 0.01), (100, 150), (0, 150)]],
    9: [[(36000, 60760), (0.7, PI), (-PI, -PI + 0.01), (900, 1200), (600, 1200)]],
}

def sample_region(k, n):
    """n uniform points in property k's region, normalised like the network's inputs"""
    boxes = torch.tensor(REGIONS[k], dtype=torch.float32)  # [boxes, 5, 2]
    lo = torch.maximum(boxes[..., 0], IN_MIN)
    hi = torch.minimum(boxes[..., 1], IN_MAX)
    which = torch.randint(len(boxes), (n,))  # property 5's two boxes are equally wide
    return normalise(lo[which] + torch.rand(n, 5) * (hi[which] - lo[which]))

def region_grid(k, side):
    """boxes (lo, hi) of side about `side`, in normalised units, that together cover property k's region"""
    los, his = [], []
    for box in REGIONS[k]:
        b = torch.tensor(box, dtype=torch.float32)
        lo, hi = normalise(torch.maximum(b[:, 0], IN_MIN)), normalise(torch.minimum(b[:, 1], IN_MAX))
        steps = [max(1, math.ceil(width / side)) for width in (hi - lo).tolist()]
        corner = torch.cartesian_prod(*[torch.arange(n) / n for n in steps])  # each box's low corner, in [0, 1)
        los.append(lo + corner * (hi - lo))
        his.append(lo + (corner + 1 / torch.tensor(steps)) * (hi - lo))
    return torch.cat(los), torch.cat(his)

# finetune the model, including all our favorite hyperparam tuning

@torch.no_grad()
def accuracy(net, test):
    """fraction of a test set (inputs, labels) on which net gives the label's advisory (the lowest score)"""
    X, labels = test
    return (net(X).argmin(1) == labels).float().mean().item()

@torch.no_grad()
def unsatisfied(net, k, lo, hi, args):
    """the boxes (or points) whose QLL bound for property k is still above -margin"""
    bound = torch.cat([qll_bound(net, k, args.p, lo[i:i + 4096], hi[i:i + 4096]) for i in range(0, len(lo), 4096)])
    return (bound > -args.margin).nonzero().squeeze(1)

def finetune(net, prop_boxes, data, props, args, tests):
    """train net on the property loss of props over their boxes (sampled points are boxes of size 0), mixed with
    args.data_weight x the data loss: the MSE to the official network's scores on data = (inputs, scores).
    Returns the test accuracies before training and after every epoch."""
    X_data, Y_data = data
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    accs = [{name: accuracy(net, test) for name, test in tests.items()}]
    for epoch in range(args.epochs):
        # hinged: a box stops pushing once its QLL bound is below -margin (raw QLL is unbounded below),
        # so each epoch only the boxes still above it are trained on
        todo = {k: unsatisfied(net, k, *prop_boxes[k], args) for k in props}
        total, n_batches = 0.0, 0
        for idx in torch.randperm(len(X_data)).split(args.batch):
            loss = args.data_weight * F.mse_loss(net(X_data[idx]), Y_data[idx])
            for k in props:
                if len(todo[k]):
                    lo, hi = prop_boxes[k]
                    b = todo[k][torch.randint(len(todo[k]), (args.batch,))]
                    loss = loss + F.relu(qll_bound(net, k, args.p, lo[b], hi[b]) + args.margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n_batches = total + loss.item(), n_batches + 1
        accs.append({name: accuracy(net, test) for name, test in tests.items()})
        print(f"epoch {epoch + 1}: loss {total / n_batches:.6g}  {sum(len(t) for t in todo.values())} not yet satisfied  "
              + "  ".join(f"{name} accuracy {acc:.4f}" for name, acc in accs[-1].items()))
    return accs

# accuracy fig, hypo: at first we tank the accuracy, save in figs/
#

def plot_accuracy(accs, name, sampling):
    for test in accs[0]:
        plt.plot(range(len(accs)), [acc[test] for acc in accs], label=f"{test} test set")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{name} ({sampling} sampling)")
    plt.legend()
    path = os.path.join(HERE, "figs", f"{name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print(f"saved {path}")

# save finetuned nets torch -> onnx,
# notation for finetuned networks: acasXu_1_1_[list of QLL properties finetuned on].onnx, i.e. acasXu_1_1_234.onnx was the official model 1_1 and was finetuned on properties 2,3, and 4
# followed by the sampling, so runs that differ only in it don't overwrite each other: acasXu_1_1_234_region.onnx

def save(net, name, settings):
    path = os.path.join(HERE, "finetuned_nets", f"{name}.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net.eval()
    # input shape [5], as acasXu.vcl's InputVector
    torch.onnx.export(net, torch.zeros(5), path, input_names=["input"], output_names=["output"], external_data=False)
    model = onnx.load(path)  # keep the run's settings in the file, verify.py copies them into results.csv
    onnx.helper.set_model_props(model, {key: str(value) for key, value in settings.items()})
    onnx.save(model, path)
    print(f"saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="official network, e.g. 1.1")
    parser.add_argument("properties", nargs="+", type=int, choices=range(10))
    parser.add_argument("--sampling", choices=["uniform", "region", "boxes"], required=True,
                        help="the property loss on points sampled uniformly, on points sampled in the property's region, "
                             "or on bounds over a grid of boxes covering the region")
    parser.add_argument("--samples", type=int, default=10000, help="points per property (uniform, region)")
    parser.add_argument("--box-size", type=float, default=0.025, help="side of the boxes (boxes), in normalised units")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--p", type=float, default=2.0, help="QLL p")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="how far below 0 a QLL value must get; QLL's soft and/or of n terms are off by "
                             "up to ln(n)/p, so a margin above that makes the property hold, not just its QLL value")
    parser.add_argument("--data", choices=["orig", "normed"], default="orig",
                        help="the training data: ART's uniform (orig) or class-balanced (normed) samples")
    parser.add_argument("--data-weight", type=float, default=0.0,
                        help="weight of the data loss, the MSE to the official network's scores on the training data")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = args.model.replace(".", "_")
    props = sorted(set(args.properties))
    name = f"acasXu_{model}_{''.join(map(str, props))}_{args.sampling}"

    net = load_onnx(os.path.join(HERE, "official_nets", f"acasXu_{model}.onnx"))
    tests = {variant: load_data(model, variant, "test") for variant in ("orig", "normed")}
    X_data = load_data(model, args.data, "train")[0]
    data = (X_data, net(X_data).detach())  # the official network's scores, the data loss's target

    prop_boxes = {}
    for k in props:
        if args.sampling == "boxes":
            prop_boxes[k] = region_grid(k, args.box_size)
        else:  # points are boxes of size 0
            X = sample_uniform(args.samples) if args.sampling == "uniform" else sample_region(k, args.samples)
            prop_boxes[k] = (X, X)

    accs = finetune(net, prop_boxes, data, props, args, tests)
    plot_accuracy(accs, name, args.sampling)
    settings = {key: value for key, value in vars(args).items() if key not in ("model", "properties")}
    settings.pop("box_size" if args.sampling != "boxes" else "samples")  # only one of them is used
    save(net, name, settings)

if __name__ == "__main__":
    main()
