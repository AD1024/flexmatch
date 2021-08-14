import tvm
import enum
from megraph.eclass import ENode
from tvm import relay
from tvm.relay import nn

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

class RelayOperatorCall(ENode):
    pass

class Symbol(ENode):
    pass

class TupleGetItem(ENode):
    pass

class ConstructTuple(ENode):
    pass

class Shape(ENode):
    pass