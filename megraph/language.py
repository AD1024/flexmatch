import tvm
import enum
from megraph.eclass import ENode
from tvm import relay
from tvm.relay import nn
from tvm.relay.op.nn.nn import global_avg_pool2d, max_pool2d

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
    RelayMinimum = relay.minimum

class AcceleratorFunc(enum.Enum):
    FlexLinear = 'flex-linear'
    FlexLSTM   = 'flex-lstm'
    VTADense   = 'vta-dense'
    VTAConv1D  = 'vta-conv1d'

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

def downcast(enode: ENode):
    symbol = enode.symbol
    lang = {
        'reshape':    RelayOperators.RelayReshape,
        'batch_norm': RelayOperators.RelayBatchNormInference,
        'softmax':    RelayOperators.RelaySoftMax,
        'relu':       RelayOperators.RelayReLU,
        'leaky_relu': RelayOperators.RelayLeakyReLU,
        'max_pool2d': RelayOperators.RelayMaxPool2D,
        'global_avg_pool2d': RelayOperators.RelayGlobalAvgPool2D,
        'avg_pool2d': RelayOperators.RelayAvgPool2D,
        'upsampling': RelayOperators.RelayUpSampling,
        'batch_flatten': RelayOperators.RelayBatchFlatten,
        'bias_add': RelayOperators.RelayBiasAdd,
        'dense':    RelayOperators.RelayDense,
        'add':      RelayOperators.RelayAdd,
        'sigmoid':  RelayOperators.RelaySigmoid,
        'minimum':  RelayOperators.RelayMinimum,
        'maximum':  RelayOperators.RelayMaximum,
    }.get(symbol, None)
    if lang is not None:
        return ENode(lang, enode.children)
    
    lang = {
        'flex-linear': AcceleratorFunc.FlexLinear,
        'flex-lstm':   AcceleratorFunc.FlexLSTM,
        'vta-dense':   AcceleratorFunc.VTADense,
        'vta-conv1d':  AcceleratorFunc.VTAConv1D,
    }.get(symbol)
    if lang is not None:
        return AcceleratorCall(lang, enode.children)
    
    return {
        'shape': lambda: Shape(enode.symbol, enode.children),
        'tuple-get-item': lambda: TupleGetItem(enode.symbol, enode.children),
        'construct-tuple': lambda: ConstructTuple(enode.symbol, enode.children),
    }.get(symbol, lambda: Symbol(enode.symbol, enode.children))()
    
