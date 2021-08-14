import tvm
from tvm import relay
from megraph.eclass import ENode
from megraph.language import *

def convert_relay_op(op):
    return {
            relay.nn.batch_norm:        RelayOperators.RelayBatchNormInference,
            relay.nn.softmax:           RelayOperators.RelaySoftMax,
            relay.nn.max_pool2d:        RelayOperators.RelayMaxPool2D,
            relay.nn.global_avg_pool2d: RelayOperators.RelayGlobalAvgPool2D,
            relay.nn.avg_pool2d:        RelayOperators.RelayAvgPool2D,
            relay.nn.upsampling:        RelayOperators.RelayUpSampling,
            relay.nn.batch_flatten:     RelayOperators.RelayBatchFlatten,
            relay.nn.bias_add:          RelayOperators.RelayBiasAdd,
            relay.nn.relu:              RelayOperators.RelayReLU,
            relay.nn.leaky_relu:        RelayOperators.RelayLeakyReLU,
            relay.reshape:              RelayOperators.RelayReshape,
            relay.nn.dense:             RelayOperators.RelayDense,
            relay.add:                  RelayOperators.RelayAdd,
            relay.maximum:              RelayOperators.RelayMaximum,
            relay.sigmoid:              RelayOperators.RelaySigmoid,
            relay.minimum:              RelayOperators.RelayMinimum
        }.get(op)