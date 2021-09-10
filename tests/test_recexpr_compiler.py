import json
import tvm
import tvm.testing
import torch
from tvm.contrib import graph_executor
from resmlp import ResMLP
from tvm import relay
from megraph.language import RecExprCompiler

def run_baseline(model, input_data):
    model.eval()
    with torch.no_grad():
        baseline_outputs = model(*[input.clone() for input in input_data])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    return baseline_outputs

def run_relay(mod, input, params):
    compiled_input = [inp.clone().cpu().numpy() for inp in input]
    target = tvm.target.create('llvm')
    vm = relay.create_executor('vm', target=target, mod=mod)
    executor = vm.evaluate()
    return executor(*compiled_input).asnumpy()

def test_compile(filename, print_model=True):
    with open(filename, 'r') as fp:
        recexpr_json = json.load(fp)
        compiler = RecExprCompiler({
            'flex-linear': 'ilaflex.linear'
        }, {
            'flex-linear': 'ilaflex'
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
        baseline = ResMLP(
            image_size = 32,
            patch_size = 16,
            dim = 64,
            depth = 3,
            num_classes = 32
        )
        if print_model:
            print(mod)

        img_input = [torch.randn(1, 3, 32, 32)]
        baseline_outputs = run_baseline(baseline, img_input)

        trace = torch.jit.trace(baseline, [input.clone() for input in img_input])
        input_names = ["input{}".format(idx) for idx, inp in enumerate(img_input)]
        input_shapes = list(zip(input_names, [inp.shape for inp in img_input]))
        _, params = relay.frontend.from_pytorch(trace, input_shapes)
        n_params = dict()
        for k, v in params.items():
            n_params[f'v{k.replace(".", "_")}'] = v
        compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in img_input]))
        
        with tvm.transform.PassContext(opt_level=3):
            for target, dev in tvm.testing.enabled_targets():
                relay_graph, lib, params = relay.build(mod, target=target, params=n_params)
                relay_model = graph_executor.create(relay_graph, lib, dev)
                relay_model.set_input(**params)
                for name, inp in compiled_input.items():
                    relay_model.set_input(name, inp)
                relay_model.run()
                for i, baseline_output in enumerate(baseline_outputs):
                    compiled_output = relay_model.get_output(i).asnumpy()
                    tvm.testing.assert_allclose(
                        baseline_output, compiled_output, rtol=1e-5, atol=1e-5
                    )

if __name__ == '__main__':
    test_compile('resmlp-dump.json')
