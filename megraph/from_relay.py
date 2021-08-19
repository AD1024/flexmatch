import tvm
from tvm import relay
from megraph.eclass import ENode
from megraph.language import *

def convert_relay_op(op):
    return {
            'nn.batch_norm':        RelayOperators.RelayBatchNormInference,
            'nn.softmax':           RelayOperators.RelaySoftMax,
            'nn.max_pool2d':        RelayOperators.RelayMaxPool2D,
            'nn.global_avg_pool2d': RelayOperators.RelayGlobalAvgPool2D,
            'nn.avg_pool2d':        RelayOperators.RelayAvgPool2D,
            'nn.upsampling':        RelayOperators.RelayUpSampling,
            'nn.batch_flatten':     RelayOperators.RelayBatchFlatten,
            'nn.bias_add':          RelayOperators.RelayBiasAdd,
            'nn.relu':              RelayOperators.RelayReLU,
            'nn.leaky_relu':        RelayOperators.RelayLeakyReLU,
            'reshape':              RelayOperators.RelayReshape,
            'nn.dense':             RelayOperators.RelayDense,
            'add':                  RelayOperators.RelayAdd,
            'maximum':              RelayOperators.RelayMaximum,
            'sigmoid':              RelayOperators.RelaySigmoid,
            'minimum':              RelayOperators.RelayMinimum
        }.get(op.name)