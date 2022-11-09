import tvm
from tvm import relay

import re

from gluoncv.model_zoo import get_model
from tvm import autotvm
import numpy as np
import vta
from vta.testing import simulator
from vta.top import graph_pack
import torchvision
import torch

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
        super().visit_call(call)
        if call.op.name == "nn.conv2d":
            print(call.args[0].checked_type.shape, call.args[1].checked_type.shape)
            self.sizes.append((call.args[0].checked_type.shape, call.args[1].checked_type.shape))
            self.paddings.append(call.attrs['padding'])
        


class DenseExtractor(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.sizes = []
    
    def visit_call(self, call):
        if call.op.name == "nn.dense":
            self.sizes.append((call.args[0].checked_type.shape, call.args[1].checked_type.shape))
        super().visit_call(call)


def profile_dense(model, filename):
    # assume a relay model here
    model = relay.transform.InferType()(model)
    size_extractor = DenseExtractor()
    size_extractor.visit(model['main'])
    sizes = size_extractor.sizes
    run_sizes = []
    results = []
    memo = {}
    for data_size, weight_size in sizes:
        if data_size[1] % 16 == 0 and weight_size[0] % 16 == 0:
            print('Running on', data_size, weight_size)
            k = f'{data_size} * {weight_size}'
            if k in memo:
                print('cached {}'.format(memo[k]))
                run_sizes.append((data_size, weight_size))
                results.append(memo[k])
                continue
            call = subprocess.run(["python3", "vta_tsim_dense.py",
                            str(data_size[0]),  # batch
                            str(data_size[1]),  # in_feat
                            str(weight_size[0])],  # out_feat
                            stdout=subprocess.PIPE)
            output = call.stdout.decode('utf-8')
            print(output)
            matched = re.match(r'(\d+)', output)
            if matched:
                print(f"Running {data_size} * {weight_size} took {int(output)}")
                run_sizes.append((data_size, weight_size))
                results.append(int(matched.group(1)))
                memo[k] = int(matched.group(1))
            else:
                print(f"failed to exec: {data_size} * {weight_size}")
    with open(filename, 'w') as f:
        for (data_size, weight_size) in run_sizes:
            f.write(f"{data_size}x{weight_size}\n")
        for result in results:
            f.write(f"{result}\n")


def profile_perf(model, backend='mxnet', filename='conv2d_perf.txt'):
    if backend == 'mxnet':
        mod, params = relay.frontend.from_mxnet(model, {"data": (1, 3, 32, 32)})
    elif backend == 'pytorch':
        torch_trace = torch.jit.trace(model, torch.rand(1, 3, 32, 32))
        mod, params = relay.frontend.from_pytorch(torch_trace, [("data", (1, 3, 32, 32))])
    model = relay.transform.InferType()(mod)
    model = relay.transform.SimplifyInference()(model)
    # print(model)
    size_extractor = Extractor("nn.conv2d")
    size_extractor.visit(model['main'])
    sizes = size_extractor.sizes

    for (data_size, wgt_size), padding in zip(sizes, size_extractor.paddings):
        call = subprocess.run(["python3", "vta_tsim_conv2d.py",
                        str(data_size[0]),  # N
                        str(data_size[1]),  # C
                        str(data_size[2]),  # H
                        str(data_size[3]),  # W
                        str(wgt_size[0]),   # O
                        str(wgt_size[2]),   # H
                        str(wgt_size[3]),   # W
                        str(padding[0]), str(padding[1]), str(padding[2]), str(padding[3])],
                        stdout=subprocess.PIPE)
        output = call.stdout.decode('utf-8')
        matched = re.match(r'(\d+)', output)
        print(output)
        with open(filename, "a") as f:
            if matched:
                f.write(f"{data_size}x{wgt_size} {output}")
            else:
                print(f"failed to exec: {data_size} * {wgt_size}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The model to profile.")
    args = parser.parse_args()
    if args.model == 'resnet':
        # model = get_model('cifar_resnet20_v2', pretrained=True, classes=10)
        # backend = 'mxnet'
        filename = 'resnet_dense_perf.txt'
        model = load_model('mxnet-resnet.relay')
    elif args.model == 'mobilenet':
        model = load_model('mobilenetv2-rewritten.relay')
        # model.eval()
        # backend = 'pytorch'
        filename = 'mobilenet_dense_perf.txt'
    profile_dense(model, filename)

if __name__ == "__main__":
    main()