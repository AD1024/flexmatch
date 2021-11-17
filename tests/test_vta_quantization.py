import math
import torch
import tqdm
import tvm
import logging
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import *
from tvm.relay import nn
from tvm.runtime.ndarray import cpu
import sys

sys.setrecursionlimit(65535)

def round_up(x: Expr):
    # return cast(left_shift(const(2, 'int32'), cast(relay.log2(relay.ceil(x)), 'int32')), 'float32')
    return power(const(2, 'float32'), ceil(log2(x)))

def bound(e):
    return relay.max(e) - relay.min(e)

def quantize_dense(data, weights, nbits, max_val):
    # weights should already be quantized prior to runtime
    act = nn.dense(data, weights, out_dtype='int32')
    # Quantize back to int8, this is what happen on VTA
    # (S_act / R_act * MAX_INT8) is a power of 2
    # implement in terms of right_shift
    act = right_shift(act, nbits)
    act = clip(act, -max_val, max_val)
    return cast(act, 'int8')

# def zero_point(data, scale, max_val):
#     data_zp = const(0, 'float32') - relay.min(data) / scale
#     return cast(clip(data_zp, 0, max_val), 'int8')
class VTAQuantize(relay.ExprMutator):
    def __init__(self, calibration_data=[], ops=[], layerwise=False, nbits=8):
        super().__init__()
        self.calibration = calibration_data.copy()
        self.ops = ops
        self.MAX_VALUE = 2 ** (nbits - 1) - 1
        self.MIN_VALUE = -(2 ** (nbits - 1)) + 1
        self.counter = 0
        self.layerwise = layerwise
        self.layers = []
        self.bindings = {}
        self.nbits = nbits
    
    def get_dtype(self, x):
        if isinstance(x, relay.Var):
            return x.type_annotation.dtype
        if isinstance(x, relay.Constant):
            return x.data.dtype
        if isinstance(x, relay.Call):
            return x.checked_type.dtype
        else:
            raise Exception(f'cannot get type for {x}')
    
    def visit_var(self, var):
        return self.bindings.get(var, var)

    def visit_let(self, let):
        new_val = self.visit(let.value)
        self.bindings[let.var] = new_val
        new_body = self.visit(let.body)
        return new_body
    
    def quantize(self, data):
        scale_p = relay.max(data) / const(self.MAX_VALUE, 'float32')
        scale_n = relay.min(data) / const(self.MIN_VALUE, 'float32')
        scale = relay.maximum(scale_p, scale_n)
        # return scale, scale
        qdata = relay.round(data / scale)
        return cast(qdata, 'int8'), scale
    
    def dequantize_uniform(self, data, scale):
        # Need to implement in terms of * and >>
        return clip(cast(data, 'float32') * scale, self.MIN_VALUE, self.MAX_VALUE)

    def visit_call(self, call):
        op = call.op
        if op.name not in self.ops:
            return relay.Call(call.op, list(map(self.visit, call.args)), call.attrs, type_args=call.type_args, span=call.span)
        args = [self.visit(x) for x in call.args]
        data = args[0]
        weights = args[1]
        qdata, S_data = self.quantize(data)
        qweight, S_w = self.quantize(weights)
        if self.calibration:
            S_ref = const(self.calibration[self.counter][1], 'float32')
            self.counter += 1
        else:
            # debug use
            _, S_ref = self.quantize(nn.dense(data, weights))
        # return qdata
        qact = nn.dense(qdata, qweight, out_dtype='int32')
        S_act = S_data * S_w / S_ref
        expr = self.dequantize_uniform(qact, S_act)
        expr = cast(expr, 'int8')
        # return qdata
        if self.layerwise:
            self.layers.append(expr)
        return relay.multiply(cast(expr, 'float32'), S_ref)
        # if not self.layerwise:
        #     if len(self.calibration):
        #         R_act = const(self.calibration[self.counter][1], 'float32')
        #         self.counter += 1
        #     else:
        #         raise Exception('incomplete calibration data')
        # else:
        #     R_act = round_up(bound(nn.dense(data, weights)))
        '''
        quant_range = const(self.MAX_VALUE, 'float32')
        data_range = bound(data)
        weights_range = bound(weights)
        rounded_up_data_range = round_up(data_range)
        rounded_up_w_range = round_up(weights_range)
        S_data = data_range / quant_range
        S_w = weights_range / quant_range
        S_act = S_data * S_w
        q_data = cast(clip(data / rounded_up_data_range * quant_range, -self.MAX_VALUE + 1, self.MAX_VALUE - 1), 'int8')
        q_weights = cast(clip(weights / rounded_up_w_range * quant_range, -self.MAX_VALUE + 1, self.MAX_VALUE - 1), 'int8')
        # NOTE: R_act need to be estimzated using a calibration set
        factor = S_act / R_act * quant_range
        # nbits = -cast(floor(log2(factor)), 'int32')
        nbits = const(self.nbits, 'int32')
        S_data_inv = rounded_up_data_range / quant_range
        S_w_inv = rounded_up_w_range / quant_range
        expr = quantize_dense(q_data, q_weights, nbits, self.MAX_VALUE - 1)
        expr = left_shift(cast(expr, 'int32'), nbits)
        expr = cast(expr, 'float32')
        expr = multiply(expr, S_data_inv * S_w_inv)
        '''
        # if self.layerwise:
        #     self.layers.append(expr)
        # return expr

class CalibrationMutator():
    def __init__(self, ops):
        class Counter(relay.ExprMutator):
            def __init__(self, ops):
                super().__init__()
                self.aggregate = []
                self.aggregate_names = []
                self.bindings = {}
                self.ops = ops
            
            def reset(self):
                self.aggregate = []
                self.aggregate_names = []
    
            def visit_call(self, call):
                op = call.op
                args = [self.visit(x) for x in call.args]
                args = list(map(lambda x: self.bindings.get(x, x) if isinstance(x, Var) else x, args))
                if op.name in self.ops:
                    self.aggregate_names.append(op.name)
                    self.aggregate.append(Call(op, args, call.attrs, call.type_args, call.span))
                return Call(op, args, call.attrs, call.type_args, call.span)
            
            def visit_var(self, var):
                if var in self.bindings:
                    return self.bindings[var]
                return var
            
            def visit_let(self, let):
                new_var = self.visit(let.var)
                new_val = self.visit(let.value)
                self.bindings[new_var] = new_val
                new_body = self.visit(let.body)
                return Let(new_var, new_val, new_body)

            def visit(self, expr):
                return super().visit(expr)

        self.counter = Counter(ops)
    
    def calibrate_mode(self, mod):
        self.counter.reset()
        expr = mod['main']
        self.counter.visit(expr)
        return tvm.ir.IRModule.from_expr(Tuple(self.counter.aggregate)), self.counter.aggregate_names

def get_test_workload():
    data = relay.var('data', type_annotation=relay.TensorType((4, 4)))
    weights_1 = relay.const(np.random.random((2, 4)) * 1)
    weights_2 = relay.const(np.random.random((3, 2)) * 1)
    a = relay.multiply(data, relay.const(4.0))
    b = relay.nn.dense(a, weights_1)
    c = relay.nn.relu(b)
    d = relay.nn.dense(c, weights_2)
    e = relay.nn.relu(d)
    f = relay.nn.dense(e, relay.const(np.random.random((4, 3)) * 2))
    g = relay.nn.relu(f)
    return tvm.ir.IRModule.from_expr(g)

def get_model(src):
    # Prepare a model
    with open(src) as fp:
        src = fp.read()
        return tvm.parser.fromtext(src)

def get_inputs(mod, scale):
    mod = relay.transform.InferType()(mod)
    inputs = dict()
    for var in mod['main'].params:
        shape = var.type_annotation.shape
        name_hint = var.name_hint
        inputs[name_hint] = np.random.rand(*[int(x) for x in shape]).astype('float32') * scale
    return inputs

def run_mod(mod, inputs):
    exe = build_module.create_executor('graph', mod=mod, device=cpu(0)).evaluate()
    result = exe(**inputs)
    return result

def lockstep_layerwise(mod, src, inputs):
    ref_mod, ops = CalibrationMutator(['nn.dense']).calibrate_mode(mod)
    quantizer = VTAQuantize(ops=['nn.dense'], layerwise=True)
    mod = get_model(src)
    quantizer.visit(mod['main'].body)
    expr = Tuple(quantizer.layers)
    qmod_layerwise = tvm.ir.IRModule.from_expr(expr)
    qmod_layerwise = relay.transform.InferType()(qmod_layerwise)
    for inp in inputs:
        ref_result = run_mod(ref_mod, inp)
        qmod_layer_results = run_mod(qmod_layerwise, inp)
        assert(len(ref_result) == len(qmod_layer_results))
        for (i, (ref, quant)) in enumerate(zip(ref_result, qmod_layer_results)):
            print(ref, '\n', quant)
            print(f'Layer {i} ({ops[i]}) relative error:\n', np.abs(ref.asnumpy() - quant.asnumpy()) / ref.asnumpy())

def calibrate(mod, params, repr_dataset, ops):
    def quantize(data):
        scale_p = np.max(data) / 127
        scale_n = np.min(data) / -127
        return float(scale_p) if scale_p > scale_n else float(scale_n)
    logging.info('Starting calibration')
    calibrate_mod, agg_ops = CalibrationMutator(ops).calibrate_mode(mod)
    quantizer = VTAQuantize(nbits=8)
    with tvm.transform.PassContext(opt_level=3):
        relay_graph, lib, params = relay.build(calibrate_mod, params=params, target='llvm')
        graph_rt = graph_executor.create(relay_graph, lib, device=cpu(0))
        graph_rt.set_input(**params)
        calibration_data = [list()] * len(agg_ops)
        for datum in tqdm.tqdm(repr_dataset):
            graph_rt.run(**datum)
            result = [graph_rt.get_output(i).asnumpy() for i in range(len(agg_ops))]
            for i, res in zip(range(len(calibration_data)), reversed(result)):
                calibration_data[i].append(quantize(res))
        return list(zip(agg_ops, map(np.mean, calibration_data)))

def main(src):
    mod = get_test_workload()
    inputs = [get_inputs(mod, 0.1) for _ in range(10)]
    ref_output = [run_mod(mod, inp).asnumpy() for inp in inputs]
    cali_dataset = inputs[0:len(inputs)]
    mod = relay.transform.InferType()(mod)
    calibrations = calibrate(mod, {}, cali_dataset, ['nn.dense'])
    ref = run_mod(mod, inputs[0]).asnumpy()
    expr = VTAQuantize(calibrations, ['nn.dense'], nbits=8).visit(mod['main'].body)
    qmod = tvm.ir.IRModule.from_expr(expr)
    qmod = relay.transform.InferType()(qmod)
    qmod = relay.transform.EliminateCommonSubexpr()(qmod)
    res = run_mod(qmod, inputs[0]).asnumpy()
    # print(f'quantized with nbits={8}:\n{res}')
    # print(f'rel error: {(ref - res) / ref}')
    for i, inp in enumerate(inputs):
        quant_result = run_mod(qmod, inp).asnumpy()
        print('==============================================')
        print(f'ref:\n{ref_output[i]}')
        print(f'quant:\n{quant_result}')
        print(f'Input {i} difference:\n{np.abs(ref_output[i] - quant_result) / ref_output[i]}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model under ./models')
    args = parser.parse_args()
    main(args.model)
