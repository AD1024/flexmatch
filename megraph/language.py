from functools import reduce
from numpy import random
import tvm
import enum
import resource
import sys
from megraph.eclass import ENode
from tvm import relay
from tvm.relay import TypeKind, nn
from tvm.relay.expr import Expr
from tvm.relay.op.nn.nn import global_avg_pool2d, max_pool2d
from tvm.relay.op.tensor import exp
from tvm.relay import ScopeBuilder
from typing import List, Tuple

resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(65536)


class Language:
    pass


class RelayOperators(enum.Enum):
    RelayBatchNormInference = nn.batch_norm,
    RelaySoftMax = nn.softmax,
    RelayReLU = nn.relu,
    RelayPReLU = nn.prelu,
    RelayLeakyReLU = nn.leaky_relu,
    RelayMaxPool2D = nn.max_pool2d,
    RelayGlobalAvgPool2D = nn.global_avg_pool2d,
    RelayAvgPool2D = nn.avg_pool2d,
    RelayUpSampling = nn.upsampling,
    RelayBatchFlatten = nn.batch_flatten,
    RelayBiasAdd = nn.bias_add,
    RelayDense = nn.dense,
    RelayReshape = relay.reshape,
    RelaySum = relay.sum,
    RelaySubtract = relay.subtract,
    RelayAdd = relay.add,
    RelaySigmoid = relay.sigmoid,
    RelayMaximum = relay.maximum,
    RelayMinimum = relay.minimum,
    RelayEqual = relay.equal,
    RelayMean = relay.mean,
    RelayMultiply = relay.multiply,
    RelayFixedPointMultiply = relay.fixed_point_multiply,
    RelayErf = relay.erf,
    RelayConv1D = relay.nn.conv1d,
    RelayConv2D = relay.nn.conv2d,
    RelayCast = relay.cast,
    RelayLeftShift = relay.left_shift,
    RelayRightShift = relay.right_shift,
    RelayClip = relay.clip,
    RelayTanh = relay.tanh,
    RelayRound = relay.round,
    RelayTake = relay.take,
    RelayDropout = relay.nn.dropout,
    RelayStack = relay.stack,
    RelayLogSoftmax = relay.nn.log_softmax,
    RelaySplit = relay.split,
    RelayLayerNorm = relay.nn.layer_norm,
    RelayBatchMatmul = relay.nn.batch_matmul,
    RelayStridedSlice = relay.strided_slice,
    RelayZeros = relay.zeros,
    RelayAdaptiveAvgPool2D = relay.nn.adaptive_avg_pool2d,
    RelayCopy = relay.copy,
    RelayArgMax = relay.argmax,

    def __call__(self, *x, dtype='int16'):
        # Handle special case of relay operator calls
        # could be mitigated by spliting parameters from attributes in glenside
        if self.value[0] == relay.zeros:
            return relay.zeros(shape=x[0], dtype="float32")
        if self.value[0] == relay.nn.layer_norm:
            return relay.nn.layer_norm(x[0], gamma=x[1], beta=x[2])
        if self.value[0] == relay.split:
            return relay.split(x[0], indices_or_sections=int(x[1]), axis=int(x[2])).tuple_value
        if self.value[0] == relay.stack:
            return relay.stack(x[:-1], axis=int(x[-1]))
        if self.value[0] == relay.nn.log_softmax:
            return relay.nn.log_softmax(x[0], axis=int(x[1]))
        if self.value[0] == relay.nn.dropout:
            return relay.nn.dropout_raw(x[0], rate=float(x[1]))
        if self.value[0] == relay.take:
            return relay.take(x[0], x[1], axis=int(x[2]))
        if self.value[0] == relay.nn.prelu:
            return relay.nn.prelu(x[0], alpha=float(x[1]), axis=int(x[2]))
        if self.value[0] == relay.fixed_point_multiply:
            return relay.value[0](x[0], multiplier=x[1], shift=x[2])
        if self.value[0] == relay.mean or self.value[0] == relay.sum:
            return self.value[0](x[0], axis=x[1], keepdims=int(x[2]) == 1)
        if self.value[0] == relay.subtract or self.value[0] == relay.equal or self.value[0] == relay.add or self.value[0] == relay.multiply:
            return self.value[0](x[0], x[1])
        if self.value[0] == relay.nn.bias_add:
            return self.value[0](x[0], x[1], axis=int(x[2]))
        if self.value[0] == relay.maximum or self.value[0] == relay.minimum:
            return self.value[0](x[0], x[1])
        if self.value[0] == relay.nn.batch_norm:
            return self.value[0](x[0], x[1], x[2], x[3], x[4], axis=int(x[5]), epsilon=float(x[6]))
        if self.value[0] == relay.nn.max_pool2d or self.value[0] == relay.nn.global_avg_pool2d:
            layout = x[-1]
            if layout == RelayActivationLayout.NCHW:
                if len(x) > 2:
                    return self.value[0](x[0], pool_size=x[1], strides=x[2], padding=x[3], layout='NCHW')
                else:
                    return self.value[0](x[0], layout='NCHW')
            elif layout == RelayActivationLayout.NHWC:
                return self.value[0](*x[:-1], layout='NHWC')
        if self.value[0] == relay.nn.adaptive_avg_pool2d:
            layout = x[-1]
            assert layout == RelayActivationLayout.NCHW
            return self.value[0](x[0], output_size=x[1])
        if self.value[0] == relay.argmax:
            return self.value[0](x[0], axis=x[1], keepdims=int(x[2]))
        if self.value[0] == relay.cast:
            return self.value[0](x[0], x[1].value)
        if self.value[0] == relay.nn.softmax:
            return self.value[0](x[0], axis=int(x[1]))
        if self.value[0] == relay.nn.conv2d:
            data_layout = x[-3].value
            kernel_layout = x[-2].value
            dtype = x[-1].value
            # Strides in Glenside includes the batch dimension, which is
            # not the case in relay
            # TODO: Assuming data is NCHW and kernel is OHWI
            # which means changing to another layout can break the
            # compliation (b/c of the `kernel_size` argument)
            return self.value[0](x[0], x[1], strides=tuple(x[2][1:]), padding=tuple(x[3]),
                                 groups=int(x[4]), channels=int(x[5]), kernel_size=(int(x[6][1]), int(x[6][2])),
                                 data_layout="NCHW", kernel_layout="OIHW", out_dtype=dtype)
        if self.value[0] == relay.nn.avg_pool2d:
            assert (len(x) == 5)
            data_layout = x[-1].value
            return nn.avg_pool2d(x[0], pool_size=x[1], strides=x[2], padding=x[3], layout=data_layout)

        x = list(map(lambda x: relay.const(x) if isinstance(x, float) else x, x))
        # x = list(map(lambda x: relay.const(x, dtype='int16')
        #          if isinstance(x, int) else x, x))
        try:
            return self.value[0](*x)
        except Exception as e:
            print(f'Exception caught when applying to {self.value[0]}:\n{e}')
            exit(-1)


class AcceleratorFunc(enum.Enum):
    FlexLinear = 'flex-linear'
    FlexLSTM = 'flex-lstm'
    VTADense = 'vta-dense'
    VTAConv1D = 'vta-conv1d'
    HLSCNNConv2D = 'hlscnn-conv2d'
    FlexMaxPool = 'flex-maxpool'
    NVDLALayerReLU = 'nvdla-layerrelu'
    NVDLAChannelBiasAdd = 'nvdla-channelbiasadd'
    NVDLAElemwiseMax = 'nvdla-elemwisemax'
    NVDLAConv2D = 'nvdla-conv2d'

    def __str__(self):
        return self.value


class PadType(enum.Enum):
    ZeroPadding = 'zero-padding'
    MinPadding = 'min-padding'
    IntPadding = 'int-padding'
    FloatPadding = 'float-padding'


class DType(enum.Enum):
    i16 = 'int16'
    i32 = 'int32'
    i64 = 'int64'
    f32 = 'float32'
    f64 = 'float64'
    u8 = 'uint8'


class ComputeType(enum.Enum):
    Relu = 'relu'
    Negative = 'negative'
    Sqrt = 'sqrt'
    ElementwiseDiv = 'elementwise-div'
    ElementwiseMul = 'elementwise-mul'
    ElementwiseAdd = 'elementwise-add'
    ReduceMax = 'reduce-max'


class RelayActivationLayout(enum.Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'


class RelayKernelLayout(enum.Enum):
    OIHW = 'OIHW'
    OHWI = 'OHWI'
    HWIO = 'HWIO'


class Compute(ENode):
    pass


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


class AccessConcat(ENode):
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


class AccessReshape(ENode):
    pass


class AccessWindows(ENode):
    pass


class AccessSqueeze(ENode):
    pass


class AccessPad(ENode):
    pass


class AccessFlatten(ENode):
    pass


class Access(ENode):
    pass


class PadTypeNode(ENode):
    def __init__(self, data=None):
        self.pad_data = data


class AccessPair(ENode):
    pass


class AccessInsertAxis(ENode):
    pass


class AccessSlice(ENode):
    pass


class RecExprCompiler:
    def __init__(self, composite_lib, compiler_lib, accelerator_func_lib=dict()):
        self.nodes: List[ENode] = []
        self.composite_lib = composite_lib
        self.compiler_lib = compiler_lib
        self.accelerator_func_lib = accelerator_func_lib
        self.region_counter = 0
        self._id_map = dict()
        self.let_worklist = list()
        self.worklist_set = set()
        self.access_window_memo = dict()
        self.var_count = 0

    def _load_json(self, expr_data: List[Tuple[int, str, List[int]]]):
        assert type(expr_data) == list
        for (eid, symbol, children) in expr_data:
            # Check monotonicity of eid
            # Check children node refs are valid
            assert eid == len(self.nodes)
            assert all(map(lambda x: x < len(self.nodes), children))
            if symbol in {'relay-operator-call', 'accelerator-call', 'compute'}:
                symbol = str(expr_data[children[0]][1])
                children = children[1:]
            self.nodes.append(downcast(ENode(symbol, children)))

    def next_var(self, for_, use_symbol=None, keep_record=True, shape=None):
        if for_ in self._id_map:
            return self._id_map[for_]
        self.var_count += 1
        var = relay.var(
            f'var_{self.var_count}' if use_symbol is None else use_symbol, type_annotation=shape)
        if keep_record:
            self._id_map[for_] = var
        return var

    def _to_relay_helper(self, index):
        assert index < len(self.nodes)
        if index in self._id_map:
            return self._id_map[index]
        enode = self.nodes[index]
        if isinstance(enode, RelayActivationLayout) \
                or isinstance(enode, RelayKernelLayout) \
                or isinstance(enode, DType):
            return enode
        children_exprs = list(map(self._to_relay_helper, enode.children))
        ch_vars = []
        for (ch_id, subexpr) in zip(enode.children, children_exprs):
            if isinstance(subexpr, relay.Var):
                ch_vars.append(subexpr)
            elif isinstance(subexpr, relay.Expr) or isinstance(subexpr, Access):
                if isinstance(subexpr, Access):
                    if isinstance(subexpr.symbol, relay.Var):
                        ch_vars.append(subexpr.symbol)
                        continue
                    subexpr = subexpr.symbol
                v = self.next_var(ch_id)
                if v not in self.worklist_set:
                    self.let_worklist.append((v, subexpr))
                    self.worklist_set.add(v)
                ch_vars.append(v)
            else:
                ch_vars.append(subexpr)

        if isinstance(enode, RelayOperatorCall):
            op = enode.symbol
            ch_vars = [relay.TupleGetItem(ch.astuple(), 0) if isinstance(
                ch, relay.TupleWrapper) else ch for ch in ch_vars]
            return op(*ch_vars)
        elif isinstance(enode, Access):
            if int(children_exprs[1]) == 0:
                return ch_vars[0]
            return Access(ch_vars[0], [int(children_exprs[1])])
        elif isinstance(enode, Literal):
            return float(enode.symbol)
        elif isinstance(enode, Symbol):
            if not (enode.symbol in self.input_shapes and enode.symbol in self.input_dtypes):
                raise Exception(f'{enode.symbol} is not a proper symbol')
            return self.next_var(index, use_symbol=enode.symbol,
                                 shape=relay.TensorType(self.input_shapes[enode.symbol], dtype=self.input_dtypes[enode.symbol]))
        elif isinstance(enode, Shape):
            return tuple(map(int, children_exprs))
        elif isinstance(enode, TupleGetItem):
            return relay.TupleGetItem(ch_vars[0], int(children_exprs[1]))
        elif isinstance(enode, ConstructTuple):
            return relay.Tuple(ch_vars)
        elif isinstance(enode, AccessConcat):
            return relay.concatenate([ch_vars[0], ch_vars[1]], axis=int(children_exprs[2]))
        elif isinstance(enode, AccessInsertAxis):
            if not isinstance(ch_vars[0], relay.Expr):
                ch_vars[0] = relay.const(ch_vars[0])
            return relay.expand_dims(ch_vars[0], axis=int(ch_vars[1]))
        elif isinstance(enode, AccessTensor):
            return ch_vars[-1]
        elif isinstance(enode, AccessReshape):
            return relay.reshape(ch_vars[0], children_exprs[1])
        elif isinstance(enode, AccessTranspose):
            return relay.transpose(ch_vars[0], axes=list(map(int, ch_vars[1])))
        elif isinstance(enode, AccessBroadcast):
            return ch_vars[0]
        elif isinstance(enode, AccessShape):
            return children_exprs[0]
        elif isinstance(enode, AccessFlatten):
            access_pattern = self.eclass_analysis[enode.children[0]]
            newshape = []
            if access_pattern['shape'] != []:
                newshape += [reduce(lambda x, y: x * y,
                                    access_pattern['shape'])]
            if access_pattern['item_shape'] != []:
                newshape += [reduce(lambda x, y: x * y,
                                    access_pattern['item_shape'])]
            return relay.reshape(ch_vars[0], newshape)
        elif isinstance(enode, AccessLiteral) or isinstance(enode, LiteralNode):
            return children_exprs[0]
        elif isinstance(enode, ListNode):
            return children_exprs
        elif isinstance(enode, Padding):
            return children_exprs[0]
        elif isinstance(enode, PadTypeNode):
            return enode.symbol
        elif isinstance(enode, AccessPad):
            assert isinstance(
                children_exprs[0], relay.Var) or self.eclass_analysis is not None
            if self.eclass_analysis:
                ndim = len(self.eclass_analysis[enode.children[0]]['shape']) + len(
                    self.eclass_analysis[enode.children[0]]['item_shape'])
            else:
                ndim = len(children_exprs[0].type_annotation.shape)
            assert ndim > 0
            pad_type = children_exprs[1]
            axis = int(children_exprs[2])
            pad_info = (children_exprs[3], children_exprs[4])
            try:
                # padding info of each dimension could be either int or an relay.Expr
                pad_info = tuple([int(x) for x in pad_info])
            except:
                pass
            pad_width = list()
            for i in range(ndim):
                if i == axis:
                    pad_width.append(pad_info)
                else:
                    pad_width.append((0, 0))
            if isinstance(pad_type, PadType):
                if pad_type == PadType.ZeroPadding:
                    return relay.nn.pad(ch_vars[0], pad_width, pad_value=0)
                elif pad_type == PadType.MinPadding:
                    return relay.nn.pad(ch_vars[0], pad_width, pad_value=relay.min(ch_vars[0]))
                elif pad_type == PadType.FloatPadding:
                    pad_data = self.eclass_analysis[enode.children[1]]
                    return relay.nn.pad(ch_vars[0], pad_width, pad_value=int(pad_data))
                else:
                    raise Exception(f'Unkonw PadType: {str(pad_type)}')
            else:
                raise Exception(f'Expecting a PadType, Got {type(pad_type)}')
        elif isinstance(enode, AccessSqueeze):
            return relay.squeeze(ch_vars[0], axis=[int(children_exprs[1])])
        elif isinstance(enode, AccessWindows):
            data = ch_vars[0]
            kernel_shape = children_exprs[1]
            strides = children_exprs[2]
            data_shape = self.eclass_analysis[enode.children[0]]['relay_shape']
            axis = len(data_shape) - len(kernel_shape)
            return relay.sliding_window(data, axis, kernel_shape, strides)
        elif isinstance(enode, AccessPair):
            return (ch_vars[0], ch_vars[1])
        elif isinstance(enode, AccessSlice):
            data = ch_vars[0]
            axis = int(ch_vars[1])
            start = int(ch_vars[2])
            end = int(ch_vars[3])
            return access_slice(data, self.eclass_analysis[enode.children[0]]['relay_shape'], axis, start, end)
        elif isinstance(enode, Compute):
            compute_type = enode.symbol
            for i in range(len(ch_vars)):
                if isinstance(ch_vars[i], relay.Expr):
                    continue
                if isinstance(ch_vars[i], tuple) or isinstance(ch_vars[0], list):
                    ch_vars[i] = list(map(lambda x: relay.const(
                        x) if not isinstance(x, relay.Expr) else x, ch_vars[i]))
                if isinstance(ch_vars[i], int) or isinstance(ch_vars[i], float):
                    ch_vars[i] = relay.const(ch_vars[i])

            func = {
                ComputeType.Relu: lambda: nn.relu(ch_vars[0]),
                ComputeType.Negative: lambda: relay.negative(ch_vars[0]),
                ComputeType.Sqrt: lambda: relay.sqrt(ch_vars[0]),
                ComputeType.ElementwiseMul: lambda: relay.multiply(ch_vars[0][0], ch_vars[0][1]),
                ComputeType.ElementwiseDiv: lambda: relay.divide(ch_vars[0][0], ch_vars[0][1]),
                ComputeType.ElementwiseAdd: lambda: relay.nn.bias_add(ch_vars[0][0], ch_vars[0][1]),
                ComputeType.ReduceMax: lambda: relay.max(ch_vars[0], [i for i in range(len(self.eclass_analysis[enode.children[0]]['shape']),
                                                                                       len(self.eclass_analysis[enode.children[0]]['relay_shape']))]),
            }.get(compute_type, None)
            if func:
                return func()
            else:
                raise Exception(f'Unrecognized compute type: {compute_type}')
        elif isinstance(enode, AcceleratorCall):
            func = str(enode.symbol)
            # the last argument is the output shape by convention
            assert (isinstance(children_exprs[-1], tuple))
            if self.use_debug_func:
                return self.accelerator_func_lib[func](*ch_vars[:-1])
            else:
                # In Glenside, the last parameter to accelerator-call is the inferred type
                # NB (shape not type)
                inferred_type = self.eclass_analysis[index]['relay_shape']
                accelerator_call = relay.accelerator_call(
                    func, inferred_type, out_dtype=self.out_dtypes[func])
                composite_name = self.composite_lib[func]
                compiler_name = self.compiler_lib[func]
                ch_vars = list(map(lambda x: relay.const(x) if isinstance(
                    x, float) or isinstance(x, int) else x, ch_vars))
                ch_vars = list(
                    filter(lambda x: isinstance(x, relay.Expr), ch_vars))
                inner_args = [relay.Var(f'inner_arg_{i}')
                              for i in range(len(ch_vars))]
                inner_func = relay.Function(
                    inner_args, accelerator_call, ret_type=relay.TensorType(inferred_type, dtype=self.out_dtypes[func]))
                inner_func = inner_func.with_attr("Composite", composite_name)
                outer_args = [relay.var(f'outer_arg_{i}')
                              for i in range(len(ch_vars))]
                outer_func = relay.Function(outer_args, inner_func(
                    *outer_args), ret_type=relay.TensorType(inferred_type, dtype=self.out_dtypes[func]))
                outer_func = outer_func.with_attr("Compiler", compiler_name)
                outer_func = outer_func.with_attr(
                    "Primitive", tvm.tir.IntImm("int32", 1))
                outer_func = outer_func.with_attr(
                    "global_symbol",
                    f"{composite_name}_{self.region_counter}")
                self.region_counter += 1
                return outer_func(*ch_vars)
        else:
            raise Exception(f'{type(enode)} not implemented')

    def to_relay_expr(self, expr_data, input_shapes, input_dtypes, analysis_data=dict(), out_dtypes=dict(), use_debug_func=False):
        self.region_counter = 0
        self.var_count = 0
        self._id_map.clear()
        self.let_worklist.clear()
        self.worklist_set.clear()
        self.nodes.clear()
        self._load_json(expr_data)
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.eclass_analysis = analysis_data
        self.use_debug_func = use_debug_func
        self.out_dtypes = out_dtypes
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


def _access_window(data: relay.Expr, access_dim: int, cur_dim: int, data_shape: List[int],
                   ker_shape: List[int], starts: List[int], strides: List[int]):
    ker_dim = len(ker_shape)
    if cur_dim == access_dim + ker_dim - 1:
        stacked = list()
        for i in range(0, int(data_shape[cur_dim]) - ker_shape[cur_dim - access_dim] + 1,
                       strides[cur_dim - access_dim]):
            starts[cur_dim] = i
            sliced = relay.strided_slice(data, starts,
                                         # note that in access dimension, we are only taking in 1
                                         # channel each time, so end = begin + 1
                                         # (wish a first-order operator `+`)
                                         list(
                                             map(lambda x: x + 1, starts[:access_dim]))
                                         + list(map(sum, zip(starts[access_dim:], ker_shape))))
            sliced = relay.expand_dims(sliced, access_dim)
            stacked.append(sliced)
        return relay.concatenate(stacked, access_dim)
    else:
        if cur_dim < access_dim:
            # when not reaching the compute dim, iterate
            # through each channels in this dimension
            stacked = list()
            for i in range(0, int(data_shape[cur_dim])):
                starts[cur_dim] = i
                result = _access_window(
                    data, access_dim, cur_dim + 1, data_shape, ker_shape, starts, strides)
                stacked.append(result)
            # concat outside compute dimension should be on itself
            return relay.concatenate(stacked, cur_dim)
        else:
            # at compute dimension, we slide the window
            # with strides at set at the current dimension
            stacked = list()
            for i in range(0, int(data_shape[cur_dim]) - ker_shape[cur_dim - access_dim] + 1,
                           strides[cur_dim - access_dim]):
                # set the index of current accessing dimension
                starts[cur_dim] = i
                next_dim_result = _access_window(data, access_dim, cur_dim + 1,
                                                 data_shape, ker_shape, starts, strides)
                next_dim_result = relay.expand_dims(
                    next_dim_result, access_dim)
                stacked.append(next_dim_result)
            return relay.concatenate(stacked, access_dim)


def access_window(data_shape: List[int], kernel_shape: List[int], strides: List[int]):
    assert len(kernel_shape) == len(strides)
    # data_shape = list(data.type_annotation.shape)
    # decide access / compute dims
    access_axis = len(data_shape) - len(kernel_shape)
    assert access_axis >= 0
    # begin with 0 each time (probably not, so that we could get rid of paddings?)
    starts = [0 for _ in range(len(data_shape))]
    meta_var = relay.var(
        f'access_window_var_{random.randint(0, 2**31)}', type_annotation=relay.TensorType(data_shape))
    return relay.Function([meta_var], _access_window(meta_var, access_axis, 0, data_shape, kernel_shape, starts, strides))


def access_slice(data: relay.Expr, data_shape: List[int], axis: int, begin: int, end: int):
    assert axis < len(data_shape)
    starts = [0] * len(data_shape)
    ends = data_shape.copy()
    starts[axis] = begin
    ends[axis] = end
    return relay.strided_slice(data, starts, ends)


def downcast(enode: ENode):
    symbol = enode.symbol
    lang = {
        'relay-reshape':    RelayOperators.RelayReshape,
        'relay-batch-norm-inference': RelayOperators.RelayBatchNormInference,
        'relay-softmax':    RelayOperators.RelaySoftMax,
        'relay-relu':       RelayOperators.RelayReLU,
        'relay-prelu':      RelayOperators.RelayPReLU,
        'relay-leaky-relu': RelayOperators.RelayLeakyReLU,
        'relay-max-pool2d': RelayOperators.RelayMaxPool2D,
        'relay-global-avg-pool2d': RelayOperators.RelayGlobalAvgPool2D,
        'relay-avg-pool2d': RelayOperators.RelayAvgPool2D,
        'relay-upsampling': RelayOperators.RelayUpSampling,
        'relay-batch-flatten': RelayOperators.RelayBatchFlatten,
        'relay-bias-add': RelayOperators.RelayBiasAdd,
        'relay-dense':    RelayOperators.RelayDense,
        'relay-subtract': RelayOperators.RelaySubtract,
        'relay-sum':      RelayOperators.RelaySum,
        'relay-add':      RelayOperators.RelayAdd,
        'relay-sigmoid':  RelayOperators.RelaySigmoid,
        'relay-minimum':  RelayOperators.RelayMinimum,
        'relay-maximum':  RelayOperators.RelayMaximum,
        'relay-equal':    RelayOperators.RelayEqual,
        'relay-mean':     RelayOperators.RelayMean,
        'relay-mul':      RelayOperators.RelayMultiply,
        'relay-fixed-point-multiply': RelayOperators.RelayFixedPointMultiply,
        'relay-erf':      RelayOperators.RelayErf,
        'relay-conv1d':   RelayOperators.RelayConv1D,
        'relay-conv2d':   RelayOperators.RelayConv2D,
        'relay-cast':     RelayOperators.RelayCast,
        'relay-clip':     RelayOperators.RelayClip,
        'relay-left-shift': RelayOperators.RelayLeftShift,
        'relay-right-shift': RelayOperators.RelayRightShift,
        'relay-take': RelayOperators.RelayTake,
        'relay-stack': RelayOperators.RelayStack,
        'relay-dropout': RelayOperators.RelayDropout,
        'relay-tanh': RelayOperators.RelayTanh,
        'relay-log-softmax': RelayOperators.RelayLogSoftmax,
        'relay-round': RelayOperators.RelayRound,
        'relay-split': RelayOperators.RelaySplit,
        'relay-layer-norm': RelayOperators.RelayLayerNorm,
        'relay-batch-matmul': RelayOperators.RelayBatchMatmul,
        'relay-strided-slice': RelayOperators.RelayStridedSlice,
        'relay-zeros': RelayOperators.RelayZeros,
        'relay-adaptive-avg-pool2d': RelayOperators.RelayAdaptiveAvgPool2D,
        'relay-copy': RelayOperators.RelayCopy,
        'relay-argmax': RelayOperators.RelayArgMax,
    }.get(symbol, None)
    if lang is not None:
        return RelayOperatorCall(lang, enode.children)

    lang = {
        'relu': ComputeType.Relu,
        'negative': ComputeType.Negative,
        'sqrt': ComputeType.Sqrt,
        'elementwise-div': ComputeType.ElementwiseDiv,
        'reduce-max': ComputeType.ReduceMax,
        'elementwise-mul': ComputeType.ElementwiseMul,
        'elementwise-add': ComputeType.ElementwiseAdd,
    }.get(symbol, None)
    if lang:
        return Compute(lang, enode.children)

    lang = {
        'relay-activation-layout-nchw': RelayActivationLayout.NCHW,
        'relay-activation-layout-nhwc': RelayActivationLayout.NHWC,
        'relay-kernel-layout-oihw': RelayKernelLayout.OIHW,
        'relay-kernel-layout-ohwi': RelayKernelLayout.OHWI,
        'relay-kernel-layout-hwio': RelayKernelLayout.HWIO,
    }.get(symbol, None)
    if lang:
        return lang

    lang = {
        'flex-linear': AcceleratorFunc.FlexLinear,
        'flex-lstm':   AcceleratorFunc.FlexLSTM,
        'vta-dense':   AcceleratorFunc.VTADense,
        'vta-conv1d':  AcceleratorFunc.VTAConv1D,
        'hlscnn-conv2d': AcceleratorFunc.HLSCNNConv2D,
        'flex-maxpool': AcceleratorFunc.FlexMaxPool,
        'nvdla-layerrelu': AcceleratorFunc.NVDLALayerReLU,
        'nvdla-channelbiasadd': AcceleratorFunc.NVDLAChannelBiasAdd,
        'nvdla-elemwisemax': AcceleratorFunc.NVDLAElemwiseMax,
        'nvdla-conv2d': AcceleratorFunc.NVDLAConv2D,
    }.get(symbol)
    if lang is not None:
        return AcceleratorCall(lang, enode.children)
    lang = {
        'int64': DType.i64,
        'int32': DType.i32,
        'int16': DType.i16,
        'uint8': DType.u8,
        'float32': DType.f32,
        'float64': DType.f64
    }.get(symbol)

    if lang is not None:
        return lang

    if symbol.isdigit() or is_num(symbol):
        return Literal(symbol, children=enode.children)

    return {
        'shape': lambda: Shape,
        'tuple-get-item': lambda: TupleGetItem,
        'construct-tuple': lambda: ConstructTuple,
        'access-insert-axis': lambda: AccessInsertAxis,
        'access-tensor': lambda: AccessTensor,
        'access-reshape': lambda: AccessReshape,
        'access-transpose': lambda: AccessTranspose,
        'access-broadcast': lambda: AccessBroadcast,
        'access-literal': lambda: AccessLiteral,
        'access-shape': lambda: AccessShape,
        'access-squeeze': lambda: AccessSqueeze,
        'access-pad': lambda: AccessPad,
        'access': lambda: Access,
        'access-flatten': lambda: AccessFlatten,
        'zero-padding': lambda: lambda *_: PadTypeNode(PadType.ZeroPadding),
        'min-padding': lambda: lambda *_: PadTypeNode(PadType.MinPadding),
        'access-windows': lambda: AccessWindows,
        'access-slice': lambda: AccessSlice,
        'access-pair': lambda: AccessPair,
        'access-concatenate': lambda: AccessConcat,
        'literal': lambda: LiteralNode,
        'list': lambda: ListNode
    }.get(symbol, lambda: Symbol)()(enode.symbol, enode.children)
