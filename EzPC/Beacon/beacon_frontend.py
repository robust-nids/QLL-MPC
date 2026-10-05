#%%
from torch import nn
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import grad
from torch.fx import Node 

import inspect

LOSSES = {
    "CE":     ("Softmax2",  "computeCELoss",  "getOutDer"),
    "MSE":    ("Reassign2", "computeMSELoss", "getOutDer"),
    "QLL":    ("Reassign2", "computeQLLLoss",  "getQLLOutDer"), # hard-coded QLL loss and derivatives for each operator in mul/add domains
    "TorchQLL":    ("Reassign2", "computeTorchQLLLoss",  "getTorchQLLOutDer"), # trace torch autograd graph module for loss and derivative translation to ezpc for QLL
}

class Carrier:
    def __init__(self, p):
        self.p = p
class Mul(Carrier):
    """Multiplicative Reals operators of QLL"""
    def tensor(self, a, b): return a * b
    def par(self, a, b): return a * b
    def disj(self, a, b): return (a**self.p + b**self.p)**(1/self.p)
    def conj(self, a, b): return (a**-self.p + b**-self.p)**(-1/self.p)
    def implies(self, a, b): return self.par(self.dual(a), b)
    def dual(self, a): return 1/a

class Add(Carrier):
    """Additive Reals operators of QLL"""
    def tensor(self, a, b): return a + b
    def par(self, a, b): return a + b
    def disj(self, a, b): return -(1/self.p) * torch.log(torch.exp(-a * self.p) + torch.exp(-b * self.p))
    def conj(self, a, b): return (1/self.p) * torch.log(torch.exp(a * self.p) + torch.exp(b * self.p))
    def implies(self, a, b): return self.par(self.dual(a), b)
    def dual(self, a): return -a

    def leq(self, a, b): return a - b
    def geq(self, a, b): return b - a

# Napiers Isomorphism from multiplicative domain to additive
def to_add(a): return -torch.log(a)

# Napiers Isomorphism from additive domain to multiplicative
def to_mul(a): return torch.exp(-a)

def loss(p, y0, x0, x1, label0):
    """QLL Loss for ToyNetwork 
        L = (y - x0) \\/ (y - x1) """
    q = Mul(p)
    data = y0 - label0
    return data * data + q.disj((y0 - x0).abs(), (y0 - x1).abs())

def trace_loss(p):
    """Return torch loss_gm and der_gm for loss.
        loss tree and gradient tree"""
    params = list(inspect.signature(loss).parameters)[1:] # y0, x0, x1, label0
    outputs = []
    for k, name in enumerate(params):
        if name.startswith("y"):
            outputs.append(k)
    spec = lambda *args : loss(p, *args)
    example = [torch.rand(()) for _ in params]
    loss_gm = make_fx(spec)(*example)
    der_gm = make_fx(grad(spec, argnums=tuple(outputs)))(*example)
    return params, loss_gm, der_gm

def to_ezpc_float(c):
    """torch literal to ezpc literal
    EzPC only accepts decimals with a point, i.e. 1.0, -0.4, 2.34
    Not 2, 1e-05"""
    text = f"{float(c):.12f}"
    text = text.rstrip("0")
    if text.endswith("."):
        text += "0"
    return text

# Operations whose arguments are arrays to EzPC function
ARRAY_OPS = {
    "aten.add.Tensor":        "ElemWiseAdd",
    "aten.sub.Tensor":        "ElemWiseSub",
    "aten.mul.Tensor":        "ElemWiseMul",
    "aten.div.Tensor":        "ElemWiseDiv",
    "aten.abs.default":       "AbsQLL",
    "aten.neg.default":       "ADualQLL",
    "aten.sgn.default":       "SignQLL",
    "aten.exp.default":       "Exp",
    "aten.log.default":       "Ln",
    "aten.reciprocal.default": "DualQLL",
    "aten.ones_like.default": "OnesLikeQLL",
}

# Operations of a number and array: EzPC function and constant to pass
CONST_OPS = {
    "aten.mul.Tensor": ("scalarMultiplication", lambda c: c),
    "aten.mul.Scalar": ("scalarMultiplication", lambda c: c),
    "aten.div.Tensor": ("scalarMultiplication", lambda c: 1/c),
    "aten.add.Tensor": ("AddScalarQLL", lambda c: c),
    "aten.sub.Tensor": ("AddScalarQLL", lambda c: -c),
}

def translate_node(node, names, out):
    """translate one graph node from one EzPC function
        node.target holds torch operation name
        node.args hold torch params passed to operator """
    op = str(node.target)
    args = node.args
    arr = names[args[0]]
    has_number = len(args) == 2 and not isinstance(args[1], Node)

    # convert array based operation to EzPC
    if not has_number:
        if op not in ARRAY_OPS:
            raise NotImplementedError(f"No EzPC translation for {op} for arrays")
        arrays = ", ".join(names[x] for x in args)
        return f"{ARRAY_OPS[op]}(BATCH, {arrays}, {out})"

    # handle constant and special ops
    c = args[1]
    ezpc_c = to_ezpc_float(args[1])

    if op == "aten.pow.Tensor_Scalar":
        return f"Pow(BATCH, {arr}, {ezpc_c}, {out})"
    if op == "aten.rsub.Scalar":
        return (f"ADualQLL(BATCH, {arr}, {out}) ;\n"
                f"AddScalarQLL(BATCH, {ezpc_c}, {out}, {out})")
    if op in CONST_OPS:
        function, constant = CONST_OPS[op]
        return f"{function}(BATCH, {to_ezpc_float(constant(c))}, {arr}, {out})"
    raise NotImplementedError(f"no EzPC translation for {op} with constant {c}")

SOURCES = { # "torch prefix" : "ezpc array" translation
    "y":     "fwdOut",
    "x":     "inp",
    "label": "target"}

def translate_graph(gm, params):
    """Translate a traced torch graph into EzPC statements."""
    # get args and targets
    # translate nodes
    declare, copies, body = [], [], []
    count = 0
    names = {}
    result = None

    # trace graph, node op = {placeholder, call_function, output}
    for node in gm.graph.nodes:
        if node.op == "placeholder": # formula literal
            name = params[len(declare)]
            prefix = name.rstrip("0123456789")
            names[node] = name
            declare.append(f"float_fl[BATCH] {name} ;")
            copies.append(f"{name}[i] = {SOURCES[prefix]}[i][{name[len(prefix):]}] ;")
        elif node.op == "call_function": # operator
            count += 1
            out = f"t{count}" # increment temp var
            body.append(f"float_fl[BATCH] {out} ;")
            body.append(f"{translate_node(node, names, out)} ;")
            names[node] = out
        elif node.op == "output": # ezpc array that returns the result, depends on number of classes
            value = node.args[0]
            if isinstance(value, (tuple, list)): # derivative graph: one result per output y
                result = [names[v] for v in value]
            else: # loss graph
                result = names[value]
            
    statements = declare + ["for i=[0:BATCH] {"] + copies + ["} ;"] + body
    return statements, result

def ezpc_function(fname, in_dim, out_dim, last_param, statements):
    """Wrap statements in 'def void fname(inp, taget, fwdOut, last_param){}"""
    header = (f"def void {fname}(float_fl[BATCH][{in_dim}] inp, float_fl[BATCH][{out_dim}] target, float_fl[BATCH][{out_dim}] fwdOut, {last_param})")
    return "\n".join([header + " {"] + statements + ["}"]) + "\n"

def torch_loss_to_ezpc(gm, params, fname, in_dim, out_dim):
    """Build EzPC function for computing loss
        loss: void computeTorchQLLLoss(type){body}
        inputs:
            gm: graph module that contains forward and backward operations
            fname: either computeTorchQLLLoss or getTorchQLLOutDer
            in_dim/out_dim: input/output dimension size
        outputs:
            EzPC code in structure of: (function signatures) + (body) + (ending)"""
    statements, result = translate_graph(gm, params)
    statements.append(f"getLoss(BATCH, {result}, loss) ;")
    return ezpc_function(fname, in_dim, out_dim, "float_fl[1] loss", statements)

def torch_der_to_ezpc(gm, params, fname, batch, in_dim, out_dim):
    """Derivative graph -> EzPC fucntion writing dLoss/dy"""
    statements, result = translate_graph(gm, params)
    # columns y
    # columns of der that loss() writes to: "y0" -> "0", "y5" -> "5"
    columns = []
    for name in params:
        if name.startswith("y"):
            columns.append(name[1:])
    for j, r in zip(columns, result):
        statements.append(f"float_fl[BATCH] g{j} ;")
        statements.append(f"scalarMultiplication(BATCH, {to_ezpc_float(1 / batch)}, {r}, g{j}) ;")
    
    statements += ["for i=[0:BATCH] {"] + [f"der[i][{j}] = g{j}[i] ;" for j in columns] + ["} ;"]

    return ezpc_function(fname, in_dim, out_dim, f"float_fl[BATCH][{out_dim}] der", statements)

class Layer :
    def __init__(self, in_features, out_features, layer_no) :
        assert 0 < layer_no
        assert 0 < in_features
        assert 0 < out_features

        self.in_features = in_features
        self.out_features = out_features
        self.layer_no = layer_no

    def __str__(self) :
        return f"Layer({self.in_features}, {self.out_features})"
        
    def get_dims(self) :
        return (self.in_features, self.out_features)
    
    def get_wt_type(self, transpose=False) :
        dim1, dim2 = self.get_dims()
        if transpose : dim1, dim2 = dim2, dim1
        return f"float_fl[{dim2}][{dim1}]"
    
    def get_bias_type(self) :
        _, outf = self.get_dims()
        return f"float_fl[{outf}]"
    
    def get_wt_input_stmt(self) :
        return f"input(SERVER, layer{self.layer_no}W, {self.get_wt_type()}) ; \n"
    
    def get_bias_input_stmt(self) :
        return f"input(SERVER, layer{self.layer_no}b, {self.get_bias_type()}) ; \n"

        
class Network :
    def __init__(self, in_dim, hidden_dims, no_out) :
        assert in_dim > 0
        
        self.in_dim = in_dim
        self.no_out = no_out
        all_dims = [in_dim] + hidden_dims + [no_out]
        self.layers = []
        for ind, (n1, n2) in enumerate(zip(all_dims[:-1], all_dims[1:])) :
            self.layers += [Layer(n1, n2, ind+1)]
            
    @property
    def no_class(self) :
        return self.no_out
        
    def __len__(self) :
        return len(self.layers)
        
    def __getitem__(self, n) :
        return self.layers[n]
    
    def __str__(self) :
        str1 = "Neural network of the following layers - \n"
        for l in self.layers :
            str1 += f"\t{l}\n"
        return str1
    
class BeaconTranslator :
    def __init__(self, net, batch, iters, lr, loss="CE", momentum=False, name="", qll_p=2.0) :
        self.batch = batch
        self.iters = iters
        self.net = net
        self.lr = lr
        self.loss = loss
        self.momentum = momentum
        self.name = name
        self.qll_p = float(qll_p)

    def __str__(self) :
        str1 = f"Using a batch size of {self.batch} to train for {self.iters} iterations with lr={self.lr}\n"
        str1 += str(self.net)
        return str1
    
    def get_batch_decl(self) :
        str1 = f"int32 BATCH={self.batch} ;\n"
        return str1
        
    def get_forward_header(self) :
        header_list = ""
        net_len = len(self.net)
        for ind, l in enumerate(self.net.layers) :
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            header_list += f"{wt_type} layer{ind+1}W, {bias_type} layer{ind+1}b, "
            
        header_list += f"float_fl[BATCH][{self.net.in_dim}] layer1In, "
        for ind, l in enumerate(self.net.layers[:-1]) :
            _, outf = l.get_dims()
            header_list += f"\
bool_bl[BATCH][{outf}] layer{ind+1}ReluHot, \
float_fl[BATCH][{outf}] layer{ind+1}Out, \
float_fl[BATCH][{outf}] layer{ind+2}In, "
            
        header_list += f"float_fl[BATCH][{self.net.no_class}] fwdOut"        
        header = f"def void forward({header_list})"
        return header
    
    def get_forward_body(self) :
        net_len = len(self.net)
        body = ""
        for ind, l in enumerate(self.net.layers[:-1]) :
            inf, outf = l.get_dims()
            body += f"\
{l.get_wt_type(transpose=True)} layer{ind+1}WReshaped ;\n\
float_fl[BATCH][{outf}] layer{ind+1}Temp ;\n\
Transpose({inf}, {outf}, layer{ind+1}W, layer{ind+1}WReshaped) ;\n\
MatMul(BATCH, {inf}, {outf}, layer{ind+1}In, layer{ind+1}WReshaped, layer{ind+1}Temp) ;\n\
GemmAdd(BATCH, {outf}, layer{ind+1}Temp, layer{ind+1}b, layer{ind+1}Out) ;\n\
Relu2(BATCH, {outf}, layer{ind+1}Out, layer{ind+2}In, layer{ind+1}ReluHot) ;\n\
\n"
         
        ind = net_len
        l = self.net.layers[-1]
        inf, outf = l.get_dims()
        output_line_args = f"BATCH, {outf}, layer{ind}Temp, fwdOut"
        act, _, _ = LOSSES[self.loss]
        output_line = f"{act}(BATCH, {outf}, layer{ind}Temp, fwdOut) ;\n"
        # output_line = f"Softmax2({output_line_args}) ;\n" if self.loss == "CE" else f"Reassign2({self.batch}, {self.net.no_out}, layer{ind}Temp, fwdOut) ;\n"
        body += f"\
{l.get_wt_type(transpose=True)} layer{ind}WReshaped ;\n\
float_fl[BATCH][{outf}] layer{ind}Temp ;\n\
Transpose({inf}, {outf}, layer{ind}W, layer{ind}WReshaped) ;\n\
MatMul(BATCH, {inf}, {outf}, layer{ind}In, layer{ind}WReshaped, layer{ind}Temp) ;\n\
GemmAdd(BATCH, {outf}, layer{ind}Temp, layer{ind}b, layer{ind}Temp) ;\n\
{output_line}"
        
        return body
    
    def get_forward_func(self) :
        brace_open = '{'
        brace_close = '}'
        return f"{self.get_forward_header()} {brace_open}\n\
{self.get_forward_body()}\n\
{brace_close}"
    
    def get_backward_header(self) :
        header_list = f"float_fl[BATCH][{self.net.no_class}] target, float_fl[BATCH][{self.net.no_class}] fwdOut"
        
        net_len = len(self.net)
        for ind, l in enumerate(self.net.layers) :
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            header_list += f", {wt_type} layer{ind+1}W, {bias_type} layer{ind+1}b"
            
        header_list += f", float_fl[BATCH][{self.net.in_dim}] layer1In"
        for ind, l in enumerate(self.net.layers[:-1]) :
            _, outf = l.get_dims()
            header_list += f"\
, bool_bl[BATCH][{outf}] layer{ind+1}ReluHot, \
float_fl[BATCH][{outf}] layer{ind+1}Out, \
float_fl[BATCH][{outf}] layer{ind+2}In"

        if self.momentum :
            for ind, l in enumerate(self.net.layers) :
                wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
                header_list += f", {wt_type} layer{ind+1}WMom, {bias_type} layer{ind+1}bMom"
              
        header = f"def void backward({header_list})"
        return header
    
    def get_backward_body(self) :
        net_len = len(self.net)
        backward_body = ""
        for i in range(net_len-1, -1, -1) :
            l = self.net.layers[i]
            inf, outf = l.get_dims()
            ind = i+1
            
            actDer_decl = f"float_fl[BATCH][{inf}] layer{ind-1}ActDer ;\n" if ind>1 else ''
            layer_decls = f"\
float_fl[BATCH][{outf}] layer{ind}Der ;\n\
float_fl[{inf}][BATCH] layer{ind}InReshaped ;\n\
float_fl[{inf}][{outf}] layer{ind}WDerReshaped ;\n\
float_fl[{outf}] layer{ind}bDer ;\n\
{actDer_decl}"
            
            arg1 = "BATCH"
            arg2 = outf
            arg3 = "fwdOut" if ind == net_len else f"layer{ind}ActDer"
            arg4 = "target" if ind == net_len else f"layer{ind}ReluHot"
            arg5 = f"layer{ind}Der"
            p_arg = f"{self.qll_p}, layer{ind}In," if self.loss == "QLL" and ind == net_len else ""
            arg_list = f"{arg1}, {arg2}, {p_arg}{arg3}, {arg4}, {arg5}"
            if self.loss == "TorchQLL" and ind == net_len:
                arg_list = f"layer1In, target, fwdOut, layer{ind}Der"

            arg_list = arg_list + (", true" if ind != net_len else '')
            func_name = LOSSES[self.loss][2] if ind == net_len else "IfElse2"
            
            act_call = f"{func_name}({arg_list}) ;\n"
            actDer_call = f"MatMul(BATCH, {outf}, {inf}, layer{ind}Der, layer{ind}W, layer{ind-1}ActDer) ;\n" if ind > 1 else ''
            layer_calls = f"\
{act_call}\
Transpose({inf}, BATCH, layer{ind}In, layer{ind}InReshaped) ;\n\
MatMul({inf}, BATCH, {outf}, layer{ind}InReshaped, layer{ind}Der, layer{ind}WDerReshaped) ;\n\
getBiasDer(BATCH, {outf}, layer{ind}Der, layer{ind}bDer) ;\n\
{actDer_call}"
            
            layer_body = f"{layer_decls}{layer_calls}\n"
            backward_body += layer_body
            
        trans_decls, trans_calls, update_calls = "", "", ""
        for ind, l in enumerate(self.net.layers) :
            inf, outf = l.get_dims()
            trans_decls += f"float_fl[{outf}][{inf}] layer{ind+1}WDer ; \n"
            trans_calls += f"Transpose({outf}, {inf}, layer{ind+1}WDerReshaped, layer{ind+1}WDer) ;\n"
            update_calls += f"\
updateWeights{'Momentum' if self.momentum else ''}2({outf}, {inf}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind+1}W, layer{ind+1}WDer{', layer'+str(ind+1)+'WMom' if self.momentum else ''}) ;\n\
updateWeights{'Momentum' if self.momentum else ''}({outf}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind+1}b, layer{ind+1}bDer{', layer'+str(ind+1)+'bMom' if self.momentum else ''}) ;\n"
            
        update_body = f"{trans_decls}{trans_calls}\n{update_calls}"
            
        return backward_body + update_body
            
    def get_backward_func(self) :
        brace_open, brace_close = '{', '}'
        return f"{self.get_backward_header()} {brace_open}\n\
{self.get_backward_body()}\n\
{brace_close}"
        
    def get_inputs(self) :
        str_inp = f"input(CLIENT, inp, float_fl[BATCH][{self.net.in_dim}]) ;\n"
        str_lab = f"input(CLIENT, target, float_fl[BATCH][{self.net.no_class}]) ;\n"
        str_ret = str_inp + str_lab
        
        for l in self.net.layers :
            str_ret += l.get_wt_input_stmt()
            str_ret += l.get_bias_input_stmt()
            
        return str_ret

    def get_mom_decls(self) :
        mom_str = ""
        for ind, l in enumerate(self.net.layers) :
            mom_str += f"\
float_fl[{l.out_features}][{l.in_features}] layer{ind+1}WMom ;\n\
float_fl[{l.out_features}] layer{ind+1}bMom ;\n\
"
        return mom_str
    
    def get_intermediate_decls(self) :
        decl_str = ""
        for ind, l in enumerate(self.net.layers[:-1]) :
            decl_str += f"\
    bool_bl[BATCH][{l.out_features}] layer{ind+1}ReluHot ;\n\
    float_fl[BATCH][{l.out_features}] layer{ind+1}Out ;\n\
    float_fl[BATCH][{l.out_features}] layer{ind+2}In ;\n\n"
            
        l = self.net.layers[-1]
        decl_str += f"\
    float_fl[BATCH][{self.net.no_class}] fwdOut ;\n\
    float_fl[1] loss ;\n"
        
        return decl_str
    
    def get_forward_call(self) :
        net_len = len(self.net)
        arg_list = ""
        for i in range(1, net_len+1) :
            arg_list += f"layer{i}W, layer{i}b, "
            
        arg_list += "inp, "
        for i in range(1, net_len) :
            arg_list += f"layer{i}ReluHot, layer{i}Out, layer{i+1}In, "
            
        arg_list += "fwdOut"
        return f"forward({arg_list}) ;\n"
    
    def get_loss_call(self) :
        if self.loss == "TorchQLL":
            return f"computeTorchQLLLoss(inp, target, fwdOut, loss) ;"
        p_arg = f"{self.net.in_dim}, {self.qll_p}, inp, " if self.loss == "QLL" else ""
        return f"{LOSSES[self.loss][1]}(BATCH, {self.net.no_class}, {p_arg} target, fwdOut, loss) ;\n"
    
    def get_backward_call(self) :
        net_len = len(self.net)
        arg_list = "target, fwdOut"
        
        for i in range(1, net_len+1) :
            arg_list += f", layer{i}W, layer{i}b"
            
        arg_list += f", inp"
        for i in range(1, net_len) :
            arg_list += f", layer{i}ReluHot, layer{i}Out, layer{i+1}In"

        if self.momentum :
            for i in range(1, net_len+1) :
                arg_list += f", layer{i}WMom, layer{i}bMom"
        
        return f"backward({arg_list}) ;\n"
        
    def get_training_loop(self) :
        brace_open, brace_close = '{', '}'
        return f"for i=[0:iters] {brace_open}\n\
            {self.get_intermediate_decls()}\n\
            {self.get_forward_call()}\n\
            {self.get_loss_call()}\n\
            output(ALL, loss[0]) ;\n\
            {self.get_backward_call()}\n\
        {brace_close} ;"
    
    def get_main(self) :
        brace_open, brace_close = '{', '}'
        iter_decl = f"int32_pl iters = {self.iters};\n"
        return f"\
def void main () {brace_open}\n\
{self.get_inputs()}\n\
{self.get_mom_decls() if self.momentum else ''}\n\
{iter_decl}\n\
{self.get_training_loop()}\n\
{brace_close}"

    def get_torch_qll_defs(self):
        """ Build loss and derivative trace from torch -> ezpc
            loss_gm (graph module): computeTorchQLLLoss
            der_gm: getTorchQLLOutDer"""
        _, loss_name, der_name = LOSSES["TorchQLL"]
        params, loss_gm, der_gm = trace_loss(self.qll_p)
        in_dim, out_dim = self.net.in_dim, self.net.no_class
        return (torch_loss_to_ezpc(loss_gm, params, loss_name, in_dim, out_dim) + "\n" + torch_der_to_ezpc(der_gm, params, der_name, self.batch, in_dim, out_dim))
    
    def get_whole_program(self) :
        decl = self.get_batch_decl()
        fwd = self.get_forward_func()
        back = self.get_backward_func()
        qll = self.get_torch_qll_defs() if self.loss == "TorchQLL" else ""
        return f"\
{decl}\n\n\
{qll}\n\n\
{fwd}\n\n\
{back}\n\n\
{self.get_main()}"
        

def torch_ffnn_to_network_args(torch_net) :
    assert isinstance(torch_net, nn.Module)

    input_size = None
    hidden_sizes = []
    output_size = None
    for p in [p.shape for p in torch_net.parameters()] :
        if input_size is None :
            input_size = p[1]
        else :
            if len(p) == 1 :
                output_size = p[0]
                continue

            hidden_sizes += [p[1]]

    return input_size, hidden_sizes, output_size


def get_translator(torch_net, batch, iters, lr, loss, momentum=False, name="", qll_p=2.0) :
    net_args = torch_ffnn_to_network_args(torch_net)
    net = Network(*net_args)
    return BeaconTranslator(net=net, batch=batch, iters=iters, lr=lr, loss=loss, momentum=momentum, name=name, qll_p=qll_p)


def dump_ezpc(trans) :
    with open("funcs.ezpc", 'r') as f :
        s = f.read()
        f.close()
        
    with open(f"{trans.name}.ezpc", 'w') as f :
        f.write(s + '\n' + trans.get_whole_program())
        f.close()
        
    print("EzPC file dumped")
