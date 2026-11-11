"""Translate the nn.py's network on loss.py's loss into hpmpc C++
    Traces the network's forward and backward pass as one graph module and splits it where they meet,
    traces the loss and its derivative, simplifies the graph modules and writes
    functions/generated/acasXu_generated.hpp: forward, loss, backward, update
    Run by scripts/XX.sh: python3 programs/qll/torch_to_hpmpc.py --batch 256"""

import argparse
import inspect
import math
import operator
import os

import torch
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import map_arg
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.func import functional_call, grad_and_value, vjp
from torch.utils._pytree import tree_flatten

from loss import data_loss, loss
from nn import SIZES, AcasXu

aten = torch.ops.aten
P = 1000  # the additive carrier's p
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "functions", "generated", "acasXu_generated.hpp")

# ---- what is traced ----

def bind(spec, x, y, l=None):
    """Call a per-sample loss.py spec on the batch columns its parameters name: x<i> input, y<i> score, l<i> target"""
    columns = {"x": x.unbind(1), "y": y.unbind(1), "l": None if l is None else l.unbind(1)}
    args = []
    for name in list(inspect.signature(spec).parameters)[1:]:
        prefix = name.rstrip("0123456789")
        args.append(columns[prefix][int(name[len(prefix):])])
    return spec(P, *args)

def batch_loss(Y, X, T):
    """finetune.py's loss of one step: the data term on the first rows (ART points, official scores T),
    the property term on the rest (points of property 1's region)"""
    y, yp = Y.split(T.shape[0])
    x, xp = X.split(T.shape[0])
    return bind(data_loss, x, y, T).mean() + bind(loss, xp, yp).mean()

def trace(model, batch):
    """Graph modules of the network (forward and backward in one) and of the loss and its derivative"""
    names = [name for name, _ in model.named_parameters()]
    w = [p.detach() for p in model.parameters()]

    def network(*args):
        params, X, dY = args[:len(w)], args[len(w)], args[len(w) + 1]
        Y, pullback = vjp(lambda *params: functional_call(model, dict(zip(names, params)), (X,)), *params)
        return Y, pullback(dY)

    def objective(Y, X, T):
        return grad_and_value(batch_loss)(Y, X, T)

    rows = 2 * batch  # batch data points, then batch property points
    X, Y, dY = torch.randn(rows, SIZES[0]), torch.randn(rows, SIZES[-1]), torch.randn(rows, SIZES[-1])
    T = torch.randn(batch, SIZES[-1])
    return names, w, optimize(make_fx(network)(*w, X, dY)), optimize(make_fx(objective)(Y, X, T))

def split(gm, nw):
    """The network graph's placeholders, output nodes Y and grads, its backward half (every node that
    depends on dY; the forward half is the rest) and the forward nodes the backward half reads"""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    late = set()
    for node in gm.graph.nodes:
        if node is placeholders[nw + 1] or any(x in late for x in node.all_input_nodes):
            late.add(node)
    saved = [n for n in gm.graph.nodes
             if n.op == "call_function" and n not in late and any(u in late and u.op != "output" for u in n.users)]
    output = next(n for n in gm.graph.nodes if n.op == "output")
    outputs, _ = tree_flatten(output.args[0])
    return placeholders, outputs[0], outputs[1:], late, saved

# ---- simplification ----

ADD = (aten.add.Tensor, aten.add.Scalar)
SUB = (aten.sub.Tensor, aten.sub.Scalar)
MUL = (aten.mul.Tensor, aten.mul.Scalar)
DIV = (aten.div.Tensor, aten.div.Scalar)
UNARY = {aten.exp.default: "exp", aten.log.default: "log", aten.reciprocal.default: "reciprocal"}
BINARY = {aten.mul.Tensor: "mul", aten.div.Tensor: "div", aten.logaddexp.default: "logaddexp"}
FOLD = {aten.add.Tensor: operator.add, aten.sub.Tensor: operator.sub, aten.neg.default: operator.neg,
        aten.mul.Tensor: operator.mul, aten.mul.Scalar: operator.mul,
        aten.div.Tensor: operator.truediv, aten.div.Scalar: operator.truediv}

def is_number(x):
    return isinstance(x, (int, float))

def scaled(x):
    """Write x as k * base, looking through multiplications by numbers and negations"""
    k = 1.0
    while isinstance(x, Node):
        if x.target in MUL and is_number(x.args[1]):
            k, x = k * x.args[1], x.args[0]
        elif x.target == aten.neg.default:
            k, x = -k, x.args[0]
        else:
            break
    return x, k

def simplify(g, node):
    """A cheaper replacement for node (a node or a number), or None to keep it"""
    a, t = node.args, node.target
    if t == aten.ones_like.default:
        return 1.0
    if t == aten.expand.default and is_number(a[0]):
        return a[0]
    if t in FOLD and all(is_number(x) for x in a):
        return float(FOLD[t](*a))
    if t == aten.t.default and a[0].target == aten.t.default:
        return a[0].args[0]
    if t in DIV and is_number(a[1]):
        return g.call_function(aten.mul.Tensor, (a[0], 1.0 / a[1]))
    if t in (aten.add.Tensor, *MUL) and is_number(a[0]):
        return g.call_function(t, (a[1], a[0]))
    if t in MUL and is_number(a[1]):
        base, k = scaled(a[0])
        if k * a[1] == 1.0:
            return base
        if base is not a[0]:
            return g.call_function(aten.mul.Tensor, (base, k * a[1]))
    if t == aten.pow.Tensor_Scalar and a[1] == 1.0:
        return a[0]
    if t == aten.pow.Tensor_Scalar and a[1] == 2.0:
        return g.call_function(aten.mul.Tensor, (a[0], a[0]))
    if t == aten.add.Tensor and a[0] is a[1]:
        return g.call_function(aten.mul.Tensor, (a[0], 2.0))
    if t == aten.exp.default:
        base, k = scaled(a[0])
        if base.target == aten.log.default and k == 1.0:
            return base.args[0]
        if base.target == aten.log.default and k == -1.0:
            return g.call_function(aten.reciprocal.default, (base.args[0],))
    return None

def rewrite(gm):
    """One pass of simplify over the graph, then dead code elimination"""
    g = gm.graph
    for node in list(g.nodes):
        if node.op != "call_function":
            continue
        with g.inserting_before(node):
            new = simplify(g, node)
        if new is None:
            continue
        if isinstance(new, Node):
            new.meta.setdefault("val", node.meta["val"])  # same shape as the node it replaces
            node.replace_all_uses_with(new)
        else:
            for user in list(node.users):
                user.args = map_arg(user.args, lambda x: new if x is node else x)
        g.erase_node(node)
    g.eliminate_dead_code()
    gm.recompile()
    return gm

def optimize(gm):
    """Fold numbers, drop transposes of transposes, share common subexpressions"""
    gm = rewrite(gm)
    gm = CSEPass()(gm).graph_module
    return rewrite(gm)  # CSE exposes x + x

# ---- translation to hpmpc ----

def shape(x):
    return tuple(x.meta["val"].shape)

def numel(x):
    return math.prod(shape(x))

class Emitter:
    """hpmpc statements computing some nodes of a graph module: local operations as loops,
    multiplications and comparisons as calls into hpmpc or functions/fixed_point_training.hpp"""

    def __init__(self, inputs, bits=None):
        self.names = dict(inputs)     # node -> C++ vector holding its shares
        self.bits = dict(bits or {})  # relu node -> C++ vector of its 0/1 shares, for threshold_backward
        self.transposed = {}          # 2-D t node -> its operand; copied out only if an array is needed
        self.constants = {}           # number -> C++ name of the public share
        self.lines, self.count = [], 0

    def compute(self, gm, outputs):
        """Statements for every node that outputs need, in graph order"""
        need, todo = set(), list(outputs)
        while todo:
            node = todo.pop()
            if node not in need and node not in self.names and node not in self.bits:
                need.add(node)
                todo.extend(node.all_input_nodes)
        for node in gm.graph.nodes:
            if node in need:
                self.node(node)

    def declare(self, n, prefix="t"):
        self.count += 1
        name = f"{prefix}{self.count}"
        self.lines.append(f"std::vector<A> {name}({n});")
        return name

    def emit(self, line):
        self.lines.append(line)

    def constant(self, c):
        if c not in self.constants:
            self.count += 1
            self.constants[c] = f"c{self.count}"
            self.lines.append(f"const A {self.constants[c]}(to_fx({c!r}));")
        return self.constants[c]

    def ref(self, x):
        """The C++ vector of x, copying a transpose out if x is one"""
        if x not in self.names:
            self.names[x] = self.transpose(self.transposed[x])
        return self.names[x]

    def transpose(self, x):
        rows, cols = shape(x)
        out = self.declare(rows * cols)
        self.emit(f"for (int i = 0; i < {rows}; i++) for (int j = 0; j < {cols}; j++) "
                  f"{out}[j * {rows} + i] = {self.ref(x)}[i * {cols} + j];")
        return out

    def at(self, x, n, out_shape):
        """x's share for element i of an n-element result, repeating x if it is a bias row or a scalar"""
        if numel(x) == n:
            return f"{self.ref(x)}[i]"
        s = shape(x)
        while s and s[0] == 1:
            s = s[1:]
        if out_shape[len(out_shape) - len(s):] != s:
            raise NotImplementedError(f"broadcast of {shape(x)} to {out_shape}")
        return f"{self.ref(x)}[i % {numel(x)}]"

    def full(self, x, n, out_shape):
        """A C++ vector of n shares of x, repeated if x is a bias row or a scalar"""
        if numel(x) == n:
            return self.ref(x)
        out = self.declare(n)
        self.emit(f"for (int i = 0; i < {n}; i++) {out}[i] = {self.at(x, n, out_shape)};")
        return out

    def gemm(self, a, b, out):
        """out = a b with hpmpc's GEMM, which takes b transposed: a t node's operand needs no copy"""
        (m, f), p = shape(a), shape(b)[1]
        bt = self.ref(self.transposed[b]) if b in self.transposed else self.transpose(b)
        self.emit(f"gemm({self.ref(a)}.data(), {bt}.data(), {out}.data(), {m}, {p}, {f});")

    def node(self, node):
        t, a = node.target, node.args
        if t in (aten.unbind.int, aten.split.Tensor):
            return  # a tuple: its getitems copy the parts out
        if t == aten.t.default and len(shape(a[0])) == 2:
            self.transposed[node] = a[0]
            return
        if t in (aten.view.default, aten._unsafe_view.default, aten.t.default):
            self.names[node] = self.ref(a[0])  # the same row-major shares
            return
        n, out_shape = numel(node), shape(node)
        out = self.declare(n)
        loop = f"for (int i = 0; i < {n}; i++) {out}[i] ="

        if t == operator.getitem:
            parent, j = a
            x = parent.args[0]
            if parent.target == aten.unbind.int and parent.args[1] == 1:
                rows, cols = shape(x)
                self.emit(f"for (int i = 0; i < {rows}; i++) {out}[i] = {self.ref(x)}[i * {cols} + {j}];")
            elif parent.target == aten.split.Tensor and len(parent.args) == 2:
                offset = j * parent.args[1] * (numel(x) // shape(x)[0])
                self.emit(f"std::copy({self.ref(x)}.begin() + {offset}, {self.ref(x)}.begin() + {offset + n}, "
                          f"{out}.begin());")
            else:
                raise NotImplementedError(f"getitem of {parent.format_node()}")
        elif t == aten.addmm.default:
            self.gemm(a[1], a[2], out)
            self.emit(f"{loop} {out}[i] + {self.at(a[0], n, out_shape)};")
        elif t == aten.mm.default:
            self.gemm(a[0], a[1], out)
        elif t in ADD + SUB and isinstance(a[0], Node) and isinstance(a[1], Node):
            op = "+" if t in ADD else "-"
            self.emit(f"{loop} {self.at(a[0], n, out_shape)} {op} {self.at(a[1], n, out_shape)};")
        elif t in ADD + SUB and is_number(a[1]):
            c = self.constant(a[1] if t in ADD else -a[1])
            self.emit(f"{loop} {self.at(a[0], n, out_shape)} + {c};")
        elif t in SUB and is_number(a[0]):
            c = self.constant(a[0])
            self.emit(f"{loop} {c} - {self.at(a[1], n, out_shape)};")
        elif t == aten.neg.default:
            zero = self.constant(0.0)
            self.emit(f"{loop} {zero} - {self.at(a[0], n, out_shape)};")
        elif t in MUL and is_number(a[1]):
            self.emit(f"mul_public({self.full(a[0], n, out_shape)}.data(), {a[1]!r}, {out}.data(), {n});")
        elif t in BINARY:
            x, y = self.full(a[0], n, out_shape), self.full(a[1], n, out_shape)
            self.emit(f"{BINARY[t]}({x}.data(), {y}.data(), {out}.data(), {n});")
        elif t in UNARY:
            self.emit(f"{UNARY[t]}({self.ref(a[0])}.data(), {out}.data(), {n});")
        elif t == aten.relu.default:
            bit = self.declare(n, "b")
            self.bits[node] = bit
            self.emit(f"relu({self.ref(a[0])}.data(), {out}.data(), {bit}.data(), {n});")
        elif t == aten.threshold_backward.default and a[2] == 0 and a[1] in self.bits:
            bit = self.bits[a[1]]  # the forward relu's 0/1 shares, no second comparison
            if is_number(a[0]):
                self.emit(f"relu_backward_public({a[0]!r}, {bit}.data(), {out}.data(), {n});")
            else:
                self.emit(f"relu_backward({self.full(a[0], n, out_shape)}.data(), {bit}.data(), {out}.data(), {n});")
        elif t == aten.sum.dim_IntList and list(a[1]) == [0] and len(shape(a[0])) == 2:
            rows, cols = shape(a[0])
            x = self.ref(a[0])
            self.emit(f"for (int j = 0; j < {cols}; j++) {{ {out}[j] = {x}[j]; "
                      f"for (int i = 1; i < {rows}; i++) {out}[j] = {out}[j] + {x}[i * {cols} + j]; }}")
        elif t in (aten.sum.default, aten.mean.default):
            x = self.ref(a[0])
            self.emit(f"{out}[0] = {x}[0]; for (int i = 1; i < {numel(a[0])}; i++) {out}[0] = {out}[0] + {x}[i];")
            if t == aten.mean.default:
                self.emit(f"mul_public({out}.data(), {1.0 / numel(a[0])!r}, {out}.data(), 1);")
        elif t == aten.expand.default:
            self.emit(f"{loop} {self.at(a[0], n, out_shape)};")
        elif t == aten.cat.default and (len(a) == 1 or a[1] == 0):
            offset = 0
            for x in a[0]:
                self.emit(f"std::copy({self.ref(x)}.begin(), {self.ref(x)}.end(), {out}.begin() + {offset});")
                offset += numel(x)
        elif t == aten.stack.default and a[1] == 1 and all(len(shape(x)) == 1 for x in a[0]):
            rows, cols = out_shape
            for j, x in enumerate(a[0]):
                self.emit(f"for (int i = 0; i < {rows}; i++) {out}[i * {cols} + {j}] = {self.ref(x)}[i];")
        else:
            raise NotImplementedError(f"no hpmpc translation for {node.format_node()}")
        self.names[node] = out

# ---- the generated file ----

SHARES = "Shares<Share>"
WEIGHTS = "std::vector<Shares<Share>>"

def hpmpc_function(comment, signature, emitter, tail):
    body = ["    using A = Additive_Share<DATATYPE, Share>;"] + ["    " + line for line in emitter.lines + tail]
    return "\n".join([comment, "template <typename Share>", f"static void {signature}", "{"] + body + ["}"]) + "\n"

def generate(batch, update):
    """The C++ of acasXu_generated.hpp"""
    model = AcasXu()
    names, w, network_gm, loss_gm = trace(model, batch)
    nw = len(w)
    placeholders, y, grads, late, saved = split(network_gm, nw)
    weights = {placeholders[k]: f"w[{k}]" for k in range(nw)}

    # forward: Y and what backward reads, from the weights and X
    f = Emitter({**weights, placeholders[nw]: "X"})
    f.compute(network_gm, [y] + saved)
    members, tail = [], [f"Y = {f.ref(y)};"]
    for n in saved:
        readers = [u for u in n.users if u in late]
        if any(not (u.target == aten.threshold_backward.default and u.args[1] is n and u.args[0] is not n)
               for u in readers):  # read as a value, not only as threshold_backward's mask
            members.append(n.name)
            tail.append(f"saved.{n.name} = {f.ref(n)};")
        if n in f.bits:
            members.append(f"{n.name}_bit")
            tail.append(f"saved.{n.name}_bit = {f.bits[n]};")
    forward = hpmpc_function(
        "// Y [ROWS][OUTPUTS] = the network of nn.py at w on X [ROWS][INPUTS]; saved keeps what backward reads",
        f"forward(const {WEIGHTS}& w, const {SHARES}& X, {SHARES}& Y, Saved<Share>& saved)", f, tail)

    # loss: its value and its derivative in Y
    lp = [n for n in loss_gm.graph.nodes if n.op == "placeholder"]
    l = Emitter({lp[0]: "Y", lp[1]: "X", lp[2]: "T"})
    (dy, value), _ = tree_flatten(next(n for n in loss_gm.graph.nodes if n.op == "output").args[0])
    l.compute(loss_gm, [dy, value])
    loss_fn = hpmpc_function(
        "// value = loss.py's loss of the step (data term on the first BATCH rows of Y, property term on the rest),\n"
        "// dY = its derivative in Y; T [BATCH][OUTPUTS] are the official network's scores of the data points",
        f"loss(const {SHARES}& Y, const {SHARES}& X, const {SHARES}& T, Additive_Share<DATATYPE, Share>& value, "
        f"{SHARES}& dY)", l, [f"value = {l.ref(value)}[0];", f"dY = {l.ref(dy)};"])

    # backward: the gradients from dY, the weights, X and what forward saved
    inputs = {**weights, placeholders[nw]: "X", placeholders[nw + 1]: "dY"}
    inputs.update({n: f"saved.{n.name}" for n in saved if n.name in members})
    b = Emitter(inputs, {n: f"saved.{n.name}_bit" for n in saved if f"{n.name}_bit" in members})
    b.compute(network_gm, grads)
    backward = hpmpc_function(
        "// grads[k] = the derivative of the loss in w[k], from dY = its derivative in Y",
        f"backward(const {WEIGHTS}& w, const {SHARES}& X, const Saved<Share>& saved, const {SHARES}& dY, "
        f"{WEIGHTS}& grads)", b, [f"grads[{k}] = {b.ref(g)};" for k, g in enumerate(grads)])

    sizes = ", ".join(str(p.numel()) for p in w)
    layout = ", ".join(f"{name} {list(p.shape)}" for name, p in zip(names, w))
    struct = ("// what forward keeps for backward: ReLU outputs the weight gradients read, ReLU 0/1 shares for their backward\n"
              "template <typename Share>\nstruct Saved\n{\n"
              f"    Shares<Share> {', '.join(members)};\n}};\n")
    return (f"#pragma once\n"
            f"// Generated by programs/qll/torch_to_hpmpc.py from programs/qll/nn.py and programs/qll/loss.py, do not edit.\n"
            f'#include "../fixed_point_training.hpp"\n#include <algorithm>\n#include <vector>\n\n'
            f"constexpr int BATCH = {batch};        // data points per step\n"
            f"constexpr int ROWS = 2 * BATCH;     // forward's input: BATCH data points, then BATCH property points\n"
            f"constexpr int INPUTS = {SIZES[0]};\n"
            f"constexpr int OUTPUTS = {SIZES[-1]};\n"
            f"constexpr int PARAMS = {nw};        // {layout}\n"
            f"constexpr int PARAM_SIZE[PARAMS] = {{{sizes}}};\n\n"
            f"template <typename Share>\nusing Shares = std::vector<Additive_Share<DATATYPE, Share>>;\n\n"
            + struct + "\n" + forward + "\n" + loss_fn + "\n" + backward + "\n" + update)

def optimizer(args):
    """The optimizer's state and update, calling sgd or adam in functions/fixed_point_training.hpp"""
    signature = f"static void update({WEIGHTS}& w, const {WEIGHTS}& grads, State<Share>& state, double lr)"
    if args.optimizer == "sgd":
        return ("// SGD keeps no state\ntemplate <typename Share>\nstruct State\n{\n};\n\n"
                "// w[k] -= lr * grads[k], SGD with one truncation for all tensors\n"
                f"template <typename Share>\n{signature}\n{{\n    sgd(w, grads, lr);\n}}\n")
    (b1, b2), eps, scale = args.betas, args.eps, args.grad_scale
    return ("// Adam's state: the bias-corrected moments of every tensor (empty before the first step) and the step count\n"
            "template <typename Share>\nstruct State\n{\n    std::vector<Shares<Share>> m, v;\n    int t = 0;\n};\n\n"
            f"// w[k] -= lr * m^[k] / (sqrt(v^[k]) + eps), Adam with betas ({b1!r}, {b2!r}) and eps {eps!r},\n"
            f"// computed on 2^{scale} grads (and eps)\n"
            f"template <typename Share>\n{signature}\n{{\n"
            f"    adam(w, grads, state.m, state.v, ++state.t, lr, {b1!r}, {b2!r}, {eps!r}, {scale});\n}}\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=256, help="data points per step (as many property points again)")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], help="Adam's b1 and b2")
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="Adam's eps, at least 2^(-FRACTIONAL/2 - grad-scale) (torch's 1e-8 rounds to 0)")
    parser.add_argument("--grad-scale", type=int, default=5,
                        help="Adam runs on 2^grad-scale grads: |2^grad-scale g| must stay below 2^(NORM_MAX/2), 1448 at 64/20")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    code = generate(args.batch, optimizer(args))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as file:
        file.write(code)
    print(f"wrote {args.out}: {code.count(chr(10))} lines")

if __name__ == "__main__":
    main()