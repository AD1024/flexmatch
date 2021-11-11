# This model is taken from https://github.com/mnikitin/EfficientNet.git
import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from math import ceil
import tvm
import subprocess
from tvm import relay
from tvm.relay.op.contrib import ilacnn

# Test code taken from 3la-tvm (https://github.com/uwsampl/3la-tvm)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ENET_DIR = os.path.join(TEST_DIR, "params")
PARAMS_FILE = os.path.join(ENET_DIR, "0.3358-imagenet-efficientnet-b0-47-best.params")

def data_present():
    return os.path.exists(PARAMS_FILE)

def get_data():
    if not os.path.isdir(ENET_DIR):
        os.mkdir(ENET_DIR)
    subprocess.run(["wget", "https://www.dropbox.com/s/l2ehu85vmmj3w5w/0.3358-imagenet-efficientnet-b0-47-best.params"], cwd=ENET_DIR)

class ReLU6(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, lite=False):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True, momentum=0.99, epsilon=1e-3))
    if active:
        if lite:
            out.add(ReLU6())
        else:
            out.add(nn.Swish())


class MBConv(nn.HybridBlock):
    def __init__(self, in_channels, channels, t, kernel, stride, lite, **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()
            _add_conv(self.out, in_channels * t, active=True, lite=lite)
            _add_conv(self.out, in_channels * t, kernel=kernel, stride=stride,
                      pad=int((kernel-1)/2), num_group=in_channels * t,
                      active=True, lite=lite)
            _add_conv(self.out, channels, active=False, lite=lite)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class EfficientNet(nn.HybridBlock):
    r"""
    Parameters
    ----------
    alpha : float, default 1.0
        The depth multiplier for controling the model size. The actual number of layers on each channel_size level
        is equal to the original number of layers multiplied by alpha.
    beta : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by beta.
    dropout_rate : float, default 0.0
        Dropout probability for the final features layer.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, alpha=1.0, beta=1.0, lite=False,
                 dropout_rate=0.0, classes=1000, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                # stem conv
                channels = 32 if lite else int(32 * beta)
                _add_conv(self.features, channels, kernel=3, stride=2, pad=1,
                          active=True, lite=lite)

                # base model settings
                repeats = [1, 2, 2, 3, 3, 4, 1]
                channels_num = [16, 24, 40, 80, 112, 192, 320]
                kernels_num = [3, 3, 5, 3, 5, 5, 3]
                t_num = [1, 6, 6, 6, 6, 6, 6]
                strides_first = [1, 2, 2, 1, 2, 2, 1]

                # determine params of MBConv layers
                in_channels_group = []
                for rep, ch_num in zip([1] + repeats[:-1], [32] + channels_num[:-1]):
                    in_channels_group += [int(ch_num * beta)] * int(ceil(alpha * rep))
                channels_group, kernels, ts, strides = [], [], [], []
                for rep, ch, kernel, t, s in zip(repeats, channels_num, kernels_num, t_num, strides_first):
                    rep = int(ceil(alpha * rep))
                    channels_group += [int(ch * beta)] * rep
                    kernels += [kernel] * rep
                    ts += [t] * rep
                    strides += [s] + [1] * (rep - 1)

                # add MBConv layers
                for in_c, c, t, k, s in zip(in_channels_group, channels_group, ts, kernels, strides):
                    self.features.add(MBConv(in_channels=in_c, channels=c, t=t, kernel=k,
                                             stride=s, lite=lite))

                # head layers
                last_channels = int(1280 * beta) if not lite and beta > 1.0 else 1280
                _add_conv(self.features, last_channels, active=True, lite=lite)
                self.features.add(nn.GlobalAvgPool2D())

            # features dropout
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

            # output layer
            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                    nn.Flatten()
                )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.output(x)
        return x


def get_efficientnet(model_name, num_classes=1000):
    params_dict = { # (width_coefficient, depth_coefficient, input_resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }
    width_coeff, depth_coeff, input_resolution, dropout_rate = params_dict[model_name]
    model = EfficientNet(alpha=depth_coeff, beta=width_coeff, lite=False,
                         dropout_rate=dropout_rate, classes=num_classes)
    return model, input_resolution


def get_efficientnet_lite(model_name, num_classes=1000):
    params_dict = { # (width_coefficient, depth_coefficient, input_resolution, dropout_rate)
        'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
        'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
        'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
        'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
        'efficientnet-lite4': (1.4, 1.8, 300, 0.3)
    }
    width_coeff, depth_coeff, input_resolution, dropout_rate = params_dict[model_name]
    model = EfficientNet(alpha=depth_coeff, beta=width_coeff, lite=True,
                         dropout_rate=dropout_rate, classes=num_classes)
    return model, input_resolution

def get_relay_mxnet(model_name, input_shape, filename):
    if not data_present():
        get_data()
    model, _ = get_efficientnet(model_name)
    model.load_parameters(PARAMS_FILE)
    mod, params = relay.frontend.from_mxnet(model, {'data': input_shape})
    mod['main'] = ilacnn.remove_padding(mod['main'])
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.SimplifyInference()(mod)
    with open(os.path.join(os.environ['FLEXMATCH_HOME'], 'tests', 'models', filename), 'w') as fp:
        fp.write(mod.astext())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of the model; one of the following:\n{}'
                                                                .format('\t'.join(['efficientnet-b{}'.format(i) for i in range(0, 8)])))
    parser.add_argument('--input-shape', type=tuple, required=False, help='input shape; optional', default=(1, 3, 224, 224))
    parser.add_argument('--filename', type=str, required=False, help='file save to', default='efficientnet.relay')

    args = parser.parse_args()
    get_relay_mxnet(args.model_name, args.input_shape, args.filename)