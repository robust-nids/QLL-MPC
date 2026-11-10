"""Convert an ACAS Xu or NIDS network between onnx and the weight files of programs/acasXu.hpp and programs/NIDS.hpp
    python3 programs/torch/onnx_converter.py to_hpmpc acasXu_2_9.onnx weights.inp
    python3 programs/torch/onnx_converter.py to_onnx finetuned.inp acasXu_2_9.onnx acasXu_2_9_1_mpc.onnx --meta lr=1e-4
    python3 programs/torch/onnx_converter.py check acasXu_2_9_1_mpc.onnx finetuned.inp
    python3 programs/torch/onnx_converter.py check acasXu_2_9.onnx acasXu_2_9_1_mpc.onnx
    with --net nids for NIDS-Verify's networks, e.g. to_onnx --net nids finetuned.inp unsat.onnx unsat_mpc.onnx"""

import argparse

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
from torch.func import functional_call

from nn import AcasXu

# nn.py's network for --net, and the range check draws its random inputs from
NETS = {"acasxu": (AcasXu, (-0.5, 0.5))}

def shapes(net="acasxu"):
    """nn.py's parameter shapes, in named_parameters() order"""
    return [tuple(p.shape) for _, p in NETS[net][0]().named_parameters()]

def linear_layers(graph):
    """The initializers of the graph, and [weight name, whether onnx stores it [in][out], bias name] of every
    linear layer in graph order: MatMul then Add (the official networks and finetune.py's), or Gemm"""
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    layers = []
    for node in graph.node:
        consts = [name for name in node.input if name in init]
        if node.op_type == "Sub" and consts:  # the official networks subtract input_AvgImg, which is zero
            if np.any(init[consts[0]]):
                raise ValueError(f"the network subtracts a nonzero {consts[0]}: fold it into the first bias first")
        elif node.op_type == "MatMul":
            layers.append([consts[0], True, None])
        elif node.op_type == "Add" and consts and layers and layers[-1][2] is None:
            layers[-1][2] = consts[0]
        elif node.op_type == "Gemm":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0 or attrs.get("transA", 0):
                raise ValueError("no conversion for a Gemm with alpha, beta or transA")
            layers.append([node.input[1], not attrs.get("transB", 0), node.input[2]])
        elif node.op_type not in ("Relu", "Flatten", "Reshape", "Identity"):
            raise ValueError(f"no conversion for {node.op_type}")
    return init, layers

def onnx_tensors(path, net="acasxu"):
    """The network's Linear weights [out][in] and biases, in nn.py's order"""
    init, layers = linear_layers(onnx.load(path).graph)
    tensors = []
    for weight, in_out, bias in layers:
        tensors += [init[weight].T if in_out else init[weight], init[bias]]
    if [t.shape for t in tensors] != shapes(net):
        raise ValueError(f"{path} has parameters {[t.shape for t in tensors]}, nn.py's {net} has {shapes(net)}")
    return tensors

def read_inp(path, net="acasxu"):
    """The tensors of a weights file, in nn.py's order"""
    values = np.array(open(path).read().split(), dtype=np.float64)
    sizes = [int(np.prod(s)) for s in shapes(net)]
    if len(values) != sum(sizes):
        raise ValueError(f"{path} has {len(values)} values, nn.py's {net} has {sum(sizes)} parameters")
    return [part.reshape(s) for part, s in zip(np.split(values, np.cumsum(sizes)[:-1]), shapes(net))]

def to_hpmpc(onnx_path, inp_path, net="acasxu"):
    """weights.inp from an onnx network"""
    with open(inp_path, "w") as file:
        for t in onnx_tensors(onnx_path, net):
            file.writelines(f"{v:.9g}\n" for v in t.ravel())  # 9 digits: every float32 exactly

def to_onnx(inp_path, template_path, onnx_path, meta, net="acasxu"):
    """An onnx network like template_path (same graph, names and opset) with the weights of inp_path"""
    model = onnx.load(template_path)
    _, layers = linear_layers(model.graph)
    index = {t.name: i for i, t in enumerate(model.graph.initializer)}
    tensors = read_inp(inp_path, net)
    for (weight, in_out, bias), w, b in zip(layers, tensors[0::2], tensors[1::2]):
        for name, value in ((weight, w.T if in_out else w), (bias, b)):
            model.graph.initializer[index[name]].CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))
    props = {p.key: p.value for p in model.metadata_props}
    props.update(meta)  # verify.py copies sampling, lr, batch, ... into results.csv
    onnx.helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)

def run_onnx(path, inputs):
    """The onnx network's outputs on inputs, one row each, in float32 as onnx stores it"""
    model = onnx.load(path)
    names = {t.name for t in model.graph.initializer}
    x = next(i for i in model.graph.input if i.name not in names)
    shape = [d.dim_value or 1 for d in x.type.tensor_type.shape.dim]
    runtime = ReferenceEvaluator(model)
    return np.stack([runtime.run(None, {x.name: v.reshape(shape).astype(np.float32)})[0].ravel() for v in inputs])

def run_inp(path, inputs, net="acasxu"):
    """nn.py's outputs with the weights file on inputs, in float64"""
    module = NETS[net][0]().double()
    params = dict(zip([name for name, _ in module.named_parameters()], map(torch.tensor, read_inp(path, net))))
    return functional_call(module, params, (torch.tensor(inputs),)).numpy()

def check(onnx_path, other_path, net="acasxu", n=1000):
    """Largest difference between the onnx network and other_path, a weights file run through nn.py or, if it ends
    in .onnx, a second onnx network, on n random inputs"""
    lo, hi = NETS[net][1]
    inputs = np.random.default_rng(0).uniform(lo, hi, (n, shapes(net)[0][1]))
    other = run_onnx(other_path, inputs) if other_path.endswith(".onnx") else run_inp(other_path, inputs, net)
    return np.abs(run_onnx(onnx_path, inputs) - other).max()

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--net", choices=list(NETS), default="acasxu")
    p = commands.add_parser("to_hpmpc", parents=[common], help="onnx network -> weights.inp")
    p.add_argument("onnx"), p.add_argument("inp")
    p = commands.add_parser("to_onnx", parents=[common], help="finetuned.inp -> onnx network, with the graph of template")
    p.add_argument("inp"), p.add_argument("template"), p.add_argument("onnx")
    p.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="metadata, e.g. lr=1e-4")
    p = commands.add_parser("check", parents=[common],
                            help="largest difference between an onnx network and a weights file or a second onnx network")
    p.add_argument("onnx"), p.add_argument("inp", help="a weights file, or an onnx network if it ends in .onnx")
    args = parser.parse_args()
    if args.command == "to_hpmpc":
        to_hpmpc(args.onnx, args.inp, args.net)
    elif args.command == "to_onnx":
        to_onnx(args.inp, args.template, args.onnx, dict(m.split("=", 1) for m in args.meta), args.net)
    else:
        print(f"largest difference on 1000 random inputs: {check(args.onnx, args.inp, args.net):.3g}")

if __name__ == "__main__":
    main()