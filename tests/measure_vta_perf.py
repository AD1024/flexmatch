import tvm
from tvm import relay

import re

from gluoncv.model_zoo import get_model
from tvm import autotvm
import numpy as np
import vta
from vta.testing import simulator
from vta.top import graph_pack

from tvm import rpc
from tvm.contrib import utils, graph_executor
import subprocess

env = vta.get_env()
# print(env.TARGET)
target = tvm.target.vta(model='tsim')

remote = rpc.LocalSession()
ctx = remote.ext_dev(0)

def load_model(filename: str):
    """Load a model from a file."""
    with open(filename, "rb") as f:
        return tvm.parser.fromtext(f.read())
    

class Extractor(relay.ExprVisitor):
    """Extract the VTA instructions from a Relay program."""
    def __init__(self, op_name):
        super().__init__()
        self.sizes = []
        self.weights = []
        self.paddings = []
        self.op_name = op_name
    
    def visit_call(self, call):
        if call.op.name == self.op_name:
            print(call.args[0].checked_type.shape, call.args[1].checked_type.shape)
            self.sizes.append((call.args[0].checked_type.shape, call.args[1].checked_type.shape))
            self.weights.append(call.args[1])
            self.paddings.append(call.attrs['padding'])
        super().visit_call(call)


def get_dense_workload(data_size, wgt_size):
    """Get a dense workload."""
    data = relay.var("data", shape=data_size)
    wgt = relay.var("wgt", shape=wgt_size)

    out = relay.nn.dense(relay.nn.relu(data), wgt)
    return relay.nn.softmax(out)

def get_conv2d(data_size, wgt_size, padding):
    """Get a conv2d workload."""
    data = relay.var("data", shape=data_size)
    wgt = relay.var("wgt", shape=wgt_size)

    out = relay.nn.conv2d(relay.nn.max_pool2d(data), wgt, kernel_size=(wgt_size[2], wgt_size[3]), channels=wgt_size[0], padding=padding)
    return relay.nn.adaptive_avg_pool2d(out)


def profile_perf(model):
    print(model)
    mod, params = relay.frontend.from_mxnet(model, {"data": (1, 3, 32, 32)})
    model = relay.transform.InferType()(mod)
    # print(model)
    size_extractor = Extractor("nn.conv2d")
    size_extractor.visit(model['main'])
    sizes = size_extractor.sizes

    return
    for (data_size, wgt_size), padding in zip(sizes, size_extractor.paddings):
        call = subprocess.run(["python3", "vta_tsim_conv2d.py", str(data_size[0]),
                        str(data_size[1]),
                        str(data_size[2]),
                        str(data_size[3]),
                        str(wgt_size[0]), str(wgt_size[2]), str(wgt_size[3]),
                        str(padding[0]), str(padding[1]), str(padding[2]), str(padding[3])],
                        stdout=subprocess.PIPE)
        output = call.stdout.decode('utf-8')
        matched = re.match(r'\n*\D+(\d+)', output)
        with open("stats10x.txt", "a") as f:
            if matched:
                f.write(f"{data_size}x{wgt_size} {matched.group(1)}\n")
            else:
                print(f"failed to exec: {data_size} * {wgt_size}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The model to profile.")
    args = parser.parse_args()
    if args.model == 'resnet':
        model = get_model('cifar_resnet20_v2', pretrained=True, classes=10)
    profile_perf(model)

if __name__ == "__main__":
    main()