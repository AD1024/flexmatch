import torch
import tvm
import logging
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import *
from tvm.relay import nn
from tvm.runtime.ndarray import cpu

def round_up(x: Expr):
    return cast(left_shift(const(2, 'int32'), cast(relay.log2(relay.ceil(x)), 'int32')), 'float32')

def bound(e):
    return relay.max(e) - relay.min(e)

def quantize_dense(data, weights, nbits):
    # weights should already be quantized prior to runtime
    act = nn.dense(data, weights, out_dtype='int32')
    # Quantize back to int8, this is what happen on VTA
    # (S_act / R_act * MAX_INT8) is a power of 2
    # implement in terms of right_shift
    act = right_shift(act, nbits)
    act = clip(act, 0, 255)
    return cast(act, 'uint8')

class VTAQuantize(relay.ExprMutator):
    def __init__(self, calibration_data=[], ops=[]):
        super().__init__()
        self.calibration = calibration_data.copy()
        self.ops = ops
        self.MAX_VALUE = 2.0 ** 8 - 1
        self.counter = 0
    
    def get_dtype(self, x):
        if isinstance(x, relay.Var):
            return x.type_annotation.dtype
        if isinstance(x, relay.Constant):
            return x.data.dtype
        if isinstance(x, relay.Call):
            return x.checked_type.dtype
        else:
            raise Exception(f'cannot get type for {x}')

    def visit_call(self, call):
        op = call.op
        if op.name not in self.ops:
            return relay.Call(call.op, list(map(self.visit, call.args)), call.attrs, type_args=call.type_args, span=call.span)
        if len(self.calibration):
            R_act, self.calibration = const(self.calibration[0][1], 'float32'), self.calibration[1:]
        else:
            raise Exception('incomplete calibration data')
        self.counter += 1
        args = [self.visit(x) for x in call.args]
        data = args[0]
        weights = args[1]
        quant_range = const(self.MAX_VALUE, 'float32')
        data_range = bound(data)
        weights_range = bound(weights)
        rounded_up_data_rang = round_up(data_range)
        rounded_up_w_range = round_up(weights_range)
        S_data = data_range / quant_range
        S_w = weights_range / quant_range
        S_act = S_data * S_w
        q_weights = cast(clip(weights / rounded_up_w_range * quant_range, 0, 255), 'uint8')
        q_data = cast(clip(data / rounded_up_data_rang * quant_range, 0, 255), 'uint8')
        # NOTE: R_act need to be estimzated using a calibration set
        factor = S_act / R_act * quant_range
        nbits = -cast(log2(factor), 'int32')
        S_data_inv = rounded_up_data_rang / quant_range
        S_w_inv = rounded_up_w_range / quant_range
        expr = quantize_dense(q_data, q_weights, nbits)
        expr = left_shift(cast(expr, 'int32'), nbits)
        expr = cast(expr, 'float32')
        expr = multiply(expr, S_data_inv * S_w_inv)
        return expr

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

def get_model(src):
    # Prepare a model
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
    # with open(src) as fp:
    #     src = fp.read()
    #     return tvm.parser.fromtext(src)

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

def calibrate(mod, params, repr_dataset, ops):
    logging.info('Starting calibration')
    calibrate_mod, agg_ops = CalibrationMutator(ops).calibrate_mode(mod)
    with tvm.transform.PassContext(opt_level=3):
        relay_graph, lib, params = relay.build(calibrate_mod, params=params, target='llvm')
        graph_rt = graph_executor.create(relay_graph, lib, device=cpu(0))
        graph_rt.set_input(**params)
        calibration_data = [list()] * len(agg_ops)
        for datum in repr_dataset:
            graph_rt.run(**datum)
            result = [graph_rt.get_output(i).asnumpy() for i in range(len(agg_ops))]
            calibration_data = [x + [np.max(y) - np.min(y)] for x, y in zip(calibration_data, result)]
        return list(zip(agg_ops, map(np.mean, calibration_data)))    

def main(src):
    mod = get_model(src)
    inputs = [get_inputs(mod, 0.1) for _ in range(10)]
    cali_dataset = inputs[0:len(inputs):2]
    mod = relay.transform.InferType()(mod)
    calibrations = calibrate(mod, {}, cali_dataset, ['nn.dense'])
    expr = VTAQuantize(calibrations, ['nn.dense']).visit(mod['main'].body)
    qmod = tvm.ir.IRModule.from_expr(expr)
    qmod = relay.transform.InferType()(qmod)
    qmod = relay.transform.EliminateCommonSubexpr()(qmod)
    for i, inp in enumerate(inputs):
        quant_result = run_mod(qmod, inp).asnumpy()
        ref_result = run_mod(mod, inp).asnumpy()
        print('==============================================')
        print(f'ref:\n{ref_result}')
        print(f'quant:\n{quant_result}')
        print(f'Input {i} difference:\n{np.abs(ref_result - quant_result) / ref_result}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model under ./models')
    args = parser.parse_args()
    main(args.model)
