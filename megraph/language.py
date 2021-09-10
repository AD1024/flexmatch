import tvm
import enum
from megraph.eclass import ENode
from tvm import relay
from tvm.relay import TypeKind, nn
from tvm.relay.op.nn.nn import global_avg_pool2d, max_pool2d
from tvm.relay.op.tensor import exp
from tvm.relay import ScopeBuilder
from typing import List, Tuple

class Language: pass

class RelayOperators(enum.Enum):
    RelayBatchNormInference = nn.batch_norm,
    RelaySoftMax = nn.softmax,
    RelayReLU = nn.relu,
    RelayLeakyReLU = nn.leaky_relu,
    RelayMaxPool2D = nn.max_pool2d,
    RelayGlobalAvgPool2D = nn.global_avg_pool2d,
    RelayAvgPool2D = nn.avg_pool2d,
    RelayUpSampling = nn.upsampling,
    RelayBatchFlatten = nn.batch_flatten,
    RelayBiasAdd = nn.bias_add,
    RelayDense = nn.dense,
    RelayReshape = relay.reshape,
    RelayAdd = relay.add,
    RelaySigmoid = relay.sigmoid,
    RelayMaximum = relay.maximum,
    RelayMinimum = relay.minimum,
    RelayMean    = relay.mean,
    RelayMultiply = relay.multiply,
    RelayErf      = relay.erf,
    RelayConv1D   = relay.nn.conv1d,

    def __call__(self, *x):
        # print(self.value, x)
        if self.value[0] == relay.mean:
            return self.value[0](x[0], axis=int(x[1]))
        if self.value[0] == relay.nn.bias_add:
            return self.value[0](x[0], x[1], axis=int(x[2]))
        x = list(map(lambda x: relay.const(x) if isinstance(x, float) else x, x))
        return self.value[0](*x)

class AcceleratorFunc(enum.Enum):
    FlexLinear = 'flex-linear'
    FlexLSTM   = 'flex-lstm'
    VTADense   = 'vta-dense'
    VTAConv1D  = 'vta-conv1d'

    def __str__(self):
        return self.value

class RelayOperatorCall(ENode):
    pass

class AcceleratorCall(ENode):
    pass

class Symbol(ENode):
    def __str__(self):
        return f'(Symbol {super().__str__()})'

class TupleGetItem(ENode):
    pass

class ConstructTuple(ENode):
    pass

class Shape(ENode):
    pass

class Literal(ENode):
    pass

class Padding(ENode):
    pass

class AccessTensor(ENode):
    pass

class AccessTranspose(ENode):
    pass

class ListNode(ENode):
    pass

class AccessBroadcast(ENode):
    pass

class AccessLiteral(ENode):
    pass

class LiteralNode(ENode):
    pass

class AccessShape(ENode):
    pass

class RecExprCompiler:
    def __init__(self, composite_lib, compiler_lib):
        self.nodes : List[ENode] = []
        self.composite_lib = composite_lib
        self.compiler_lib = compiler_lib
        self.region_counter = 0
        self._id_map = dict()
        self.let_worklist = list()
        self.worklist_set = set()
        self.var_count = 0
    
    def _load_json(self, expr_data: List[Tuple[int, str, List[int]]]):
        assert type(expr_data) == list
        for (eid, symbol, children) in expr_data:
            # Check monotonicity of eid
            # Check children node refs are valid
            assert eid == len(self.nodes)
            assert all(map(lambda x: x < len(self.nodes), children))
            if symbol in {'relay-operator-call', 'accelerator-call'}:
                symbol = str(expr_data[children[0]][1])
                children = children[1:]
            self.nodes.append(downcast(ENode(symbol, children)))

    def next_var(self, for_, use_symbol=None, keep_record=True, shape=None):
        if for_ in self._id_map:
            return self._id_map[for_]
        self.var_count += 1
        var = relay.var(f'var_{self.var_count}' if use_symbol is None else use_symbol, type_annotation=shape)
        if keep_record:
            self._id_map[for_] = var
        return var

    def _to_relay_helper(self, index):
        assert index < len(self.nodes)
        if index in self._id_map:
            return self._id_map[index]
        enode = self.nodes[index]
        children_exprs = list(map(self._to_relay_helper, enode.children))
        ch_vars = []
        for (ch_id, subexpr) in zip(enode.children, children_exprs):
            if isinstance(subexpr, relay.Var):
                ch_vars.append(subexpr)
            elif isinstance(subexpr, relay.Expr):
                v = self.next_var(ch_id)
                if v not in self.worklist_set:
                    self.let_worklist.append((v, subexpr))
                    self.worklist_set.add(v)
                ch_vars.append(v)
            else:
                ch_vars.append(subexpr)
        
        if isinstance(enode, RelayOperatorCall):
            op = enode.symbol
            return op(*ch_vars)
        elif isinstance(enode, Literal):
            return float(enode.symbol)
        elif isinstance(enode, Symbol):
            return self.next_var(index, use_symbol=enode.symbol, shape=relay.TensorType(self.input_shapes[enode.symbol]))
        elif isinstance(enode, Shape):
            return tuple(map(int, children_exprs))
        elif isinstance(enode, TupleGetItem):
            return relay.TupleGetItem(ch_vars[0], *ch_vars[1:])
        elif isinstance(enode, ConstructTuple):
            return relay.Tuple(ch_vars)
        elif isinstance(enode, AccessTensor):
            return ch_vars[-1]
        elif isinstance(enode, AccessTranspose):
            return relay.transpose(ch_vars[0], axes=list(map(int, ch_vars[1])))
        elif isinstance(enode, AccessBroadcast):
            return children_exprs[0]
        elif isinstance(enode, AccessShape):
            return children_exprs[-1]
        elif isinstance(enode, AccessLiteral) or isinstance(enode, LiteralNode):
            return children_exprs[0]
        elif isinstance(enode, ListNode):
            return children_exprs
        elif isinstance(enode, Padding):
            return children_exprs[0]
        elif isinstance(enode, AcceleratorCall):
            func = str(enode.symbol)
            accelerator_call = relay.accelerator_call(func, children_exprs[-1])
            composite_name = self.composite_lib[func]
            compiler_name = self.compiler_lib[func]
            inner_args = [relay.Var(f'inner_arg_{i}') for i in range(len(ch_vars) - 1)]
            inner_func = relay.Function(inner_args, accelerator_call, ret_type=relay.TensorType(children_exprs[-1]))
            inner_func.with_attr("Composite", composite_name)
            outer_args = [relay.var(f'outer_arg_{i}') for i in range(len(ch_vars) - 1)]
            outer_func = relay.Function(outer_args, inner_func(*outer_args), ret_type=relay.TensorType(children_exprs[-1]))
            outer_func = outer_func.with_attr("Compiler", compiler_name)
            outer_func = outer_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
            outer_func = outer_func.with_attr(
                "global_symbol",
                f"{composite_name}_{self.region_counter}")
            self.region_counter += 1
            return outer_func(*ch_vars[:-1])
        else:
            raise Exception(f'{type(enode)} not implemented')
    
    def to_relay_expr(self, expr_data, input_shapes):
        self.region_counter = 0
        self.var_count = 0
        self._id_map.clear()
        self.let_worklist.clear()
        self.worklist_set.clear()
        self.nodes.clear()
        self._load_json(expr_data)
        self.input_shapes = input_shapes
        if len(self.nodes) == 0:
            return None
        else:
            expr = self._to_relay_helper(len(self.nodes) - 1)
            if len(self.let_worklist) == 0:
                return expr
            else:
                def construct_let(worklist):
                    if len(worklist) == 0:
                        return expr
                    else:
                        (var, e), *xs = worklist
                        return relay.Let(var, e, construct_let(xs))
            return construct_let(self.let_worklist)

def is_num(x):
    try:
        float(x)
        return True
    except:
        return False

def downcast(enode: ENode):
    symbol = enode.symbol
    lang = {
        'relay-reshape':    RelayOperators.RelayReshape,
        'relay-batch_norm': RelayOperators.RelayBatchNormInference,
        'relay-softmax':    RelayOperators.RelaySoftMax,
        'relay-relu':       RelayOperators.RelayReLU,
        'relay-leaky_relu': RelayOperators.RelayLeakyReLU,
        'relay-max_pool2d': RelayOperators.RelayMaxPool2D,
        'relay-global_avg_pool2d': RelayOperators.RelayGlobalAvgPool2D,
        'relay-avg_pool2d': RelayOperators.RelayAvgPool2D,
        'relay-upsampling': RelayOperators.RelayUpSampling,
        'relay-batch_flatten': RelayOperators.RelayBatchFlatten,
        'relay-bias-add': RelayOperators.RelayBiasAdd,
        'relay-dense':    RelayOperators.RelayDense,
        'relay-add':      RelayOperators.RelayAdd,
        'relay-sigmoid':  RelayOperators.RelaySigmoid,
        'relay-minimum':  RelayOperators.RelayMinimum,
        'relay-maximum':  RelayOperators.RelayMaximum,
        'relay-mean':     RelayOperators.RelayMean,
        'relay-mul':      RelayOperators.RelayMultiply,
        'relay-erf':      RelayOperators.RelayErf,
        'relay-conv1d':   RelayOperators.RelayConv1D,
    }.get(symbol, None)
    if lang is not None:
        return RelayOperatorCall(lang, enode.children)
    
    lang = {
        'flex-linear': AcceleratorFunc.FlexLinear,
        'flex-lstm':   AcceleratorFunc.FlexLSTM,
        'vta-dense':   AcceleratorFunc.VTADense,
        'vta-conv1d':  AcceleratorFunc.VTAConv1D,
    }.get(symbol)
    if lang is not None:
        return AcceleratorCall(lang, enode.children)
    
    if symbol.isdigit() or is_num(symbol):
        return Literal(symbol, children=enode.children)
    
    return {
        'shape': lambda: Shape,
        'tuple-get-item': lambda: TupleGetItem,
        'construct-tuple': lambda: ConstructTuple,
        'access-insert-axis': lambda: Padding,
        'access-tensor': lambda: AccessTensor,
        'access-transpose': lambda: AccessTranspose,
        'access-broadcast': lambda: AccessBroadcast,
        'access-literal':   lambda: AccessLiteral,
        'access-shape':     lambda: AccessShape,
        'literal':          lambda: LiteralNode,
        'list':             lambda: ListNode
    }.get(symbol, lambda: Symbol)()(enode.symbol, enode.children)
