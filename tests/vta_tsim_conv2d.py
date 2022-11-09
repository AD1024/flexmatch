import tvm
from tvm import relay

from tvm import autotvm
import numpy as np
import vta
from vta.testing import simulator
from vta.top import graph_pack

from tvm import rpc
from tvm.contrib import utils, graph_executor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, default=1)
parser.add_argument('c', type=int, default=32)
parser.add_argument('h', type=int, default=64)
parser.add_argument('w', type=int, default=16)
parser.add_argument('o', type=int, default=16)
parser.add_argument('kh', type=int, default=1)
parser.add_argument('kw', type=int, default=1)
parser.add_argument('padding_top', type=int, default=0)
parser.add_argument('padding_left', type=int, default=0)
parser.add_argument('padding_bottom', type=int, default=0)
parser.add_argument('padding_right', type=int, default=0)
args = parser.parse_args()
N, C, H, W, O, kH, kW = args.n, args.c, args.h, args.w, args.o, args.kh, args.kw
padding = (args.padding_top, args.padding_left, args.padding_bottom, args.padding_right)

# N, C, H, W, O, kH, kW = 1, 64, 8, 8, 64, 3, 3

# Notes:
'''
1. Make TVM with both lib_fsim and lib_tsim on
2. Make lib (run `make lib`) in vta-hw/hardware/chisel
3. run this script
'''

env = vta.get_env()
target = tvm.target.vta(model='sim')

remote = rpc.LocalSession()
ctx = remote.ext_dev(0)

def get_conv2d(n, c, o, h, w, kH, kW, padding):
    data = relay.var('data', shape=(n, c, h, w))
    weight = relay.var('weight', shape=(o, c, kH, kW))
    data_1 = relay.nn.max_pool2d(data)
    return relay.nn.adaptive_avg_pool2d(relay.nn.conv2d(data_1, weight, kernel_size=(kH, kW), channels=o, padding=padding))

with autotvm.tophub.context(target):

    dtype_dict = {'data': 'float32', 'weight': 'float32'}
    shape_dict = {'data': (N, C, H, W), 'weight': (O, C, kH, kW)}

    params = {
        # 'data': tvm.nd.array(np.random.uniform(size=(N, C, H, W)).astype('float32')),
        'weight': tvm.nd.array(np.random.uniform(size=(O, C, kH, kW)).astype('float32'))
    }

    e = get_conv2d(N, C, O, H, W, kH, kW, padding)
    model = tvm.IRModule.from_expr(e)
    model = relay.transform.InferType()(model)
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[]):
            relay_prog = relay.quantize.quantize(model, params=params)
        # Perform graph packing and constant folding for VTA target
        # do device annotation if target is intelfocl or sim
        relay_prog = graph_pack(
            relay_prog["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name="nn.max_pool2d",
            stop_name="nn.adaptive_avg_pool2d",
            device_annot=(env.TARGET == "intelfocl"),
        )

    with vta.build_config(
            opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}
        ):
        graph, lib, params = relay.build(
            relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
        )
    
    
    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib.tar"))
    remote.upload(temp.relpath("graphlib.tar"))
    lib = remote.load_module("graphlib.tar")

    # Graph runtime
    m = graph_executor.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('data', tvm.nd.array(np.random.uniform(size=(N, C, H, W)).astype('float32')))
    num = 1  # number of times we run module for a single measurement
    rep = 10  # number of measurements (we derive std dev from this)
    timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)
    simulator.clear_stats()
    timer()
    sim_stats = simulator.stats()
    for k, v in sim_stats.items():
        # Since we execute the workload many times, we need to normalize stats
        # Note that there is always one warm up run
        # Therefore we divide the overall stats by (num * rep + 1)
        print(v // (num * rep + 1))
