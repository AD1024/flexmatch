import json
import tvm
import torch
from tvm.contrib import graph_executor
from resmlp import ResMLP
from tvm import relay
from megraph.language import RecExprCompiler

def run_baseline(input_data):
    model = ResMLP(
        image_size = 32,
        patch_size = 16,
        dim = 64,
        depth = 3,
        num_classes = 32
    )
    model.eval()
    with torch.no_grad():
        baseline_outputs = model(*[input.clone() for input in input_data])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    return baseline_outputs

def run_relay(mod, input):
    compiled_input = [inp.clone().cpu().numpy() for inp in input]
    target = tvm.target.create('llvm')
    vm = relay.create_executor('vm', target=target, mod=mod)
    executor = vm.evaluate()
    return executor(*compiled_input).asnumpy()

def test_compile(filename, print_model=True):
    with open(filename, 'r') as fp:
        recexpr_json = json.load(fp)
        compiler = RecExprCompiler({
            'flex-linear': 'ilaflex'
        }, {
            'flex-linear': 'ilaflex.linear'
        })
 
        expr = compiler.to_relay_expr(recexpr_json, {
            'input0': (1, 3, 32, 32),
            'v1_weight': (64, 768),
            'v1_bias': (64,),
            'v2_0_affine_g': (1, 1, 64),
            'v2_0_affine_b': (1, 1, 64),
            'v2_0_fn_weight': (4, 4, 1),
            'v2_0_fn_bias': (4,),
            'v2_0_scale': (1, 1, 64),
            'v2_1_affine_g': (1, 1, 64),
            'v2_1_affine_b': (1, 1, 64),
            'v2_1_fn_0_weight': (256, 64),
            'v2_1_fn_0_bias': (256,),
            'v2_1_fn_2_weight': (64, 256),
            'v2_1_fn_2_bias': (64,),
            'v2_1_scale': (1, 1, 64),
            'v3_0_affine_g': (1, 1, 64),
            'v3_0_affine_b': (1, 1, 64),
            'v3_0_fn_weight': (4, 4, 1),
            'v3_0_fn_bias': (4,),
            'v3_0_scale': (1, 1, 64),
            'v3_1_affine_g': (1, 1, 64),
            'v3_1_affine_b': (1, 1, 64),
            'v3_1_fn_0_weight': (256, 64),
            'v3_1_fn_0_bias': (256,),
            'v3_1_fn_2_weight': (64, 256),
            'v3_1_fn_2_bias': (64,),
            'v3_1_scale': (1, 1, 64),
            'v4_0_affine_g': (1, 1, 64),
            'v4_0_affine_b': (1, 1, 64),
            'v4_0_fn_weight': (4, 4, 1),
            'v4_0_fn_bias': (4,),
            'v4_0_scale': (1, 1, 64),
            'v4_1_affine_g': (1, 1, 64),
            'v4_1_affine_b': (1, 1, 64),
            'v4_1_fn_0_weight': (256, 64),
            'v4_1_fn_0_bias': (256,),
            'v4_1_fn_2_weight': (64, 256),
            'v4_1_fn_2_bias': (64,),
            'v4_1_scale': (1, 1, 64),
            'v5_g': (1, 1, 64),
            'v5_b': (1, 1, 64),
            'v7_weight': (32, 64),
            'v7_bias': (32,),
        })
        mod = tvm.ir.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        if print_model:
            print(mod)

        input = torch.randn(1, 3, 32, 32)
        baseline_output = run_baseline([input])
        relay_output = run_relay(mod, [input])

        print(baseline_output)
        print('--------------------')
        print(relay_output)

        # tvm.testing.assert_allclose(
        #     baseline_output, compiled_output, rtol=rtol, atol=atol
        # )

if __name__ == '__main__':
    test_compile('resmlp-dump.json')