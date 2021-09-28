import json
import tvm
import tvm.testing
import torch
import time
import random
from tvm.contrib import graph_executor
from resmlp import ResMLP
from tvm import relay
from tvm.relay.op.contrib import ilaflex
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

def run_passes(mod):
    patterns = ilaflex.pattern_table()
    mod = tvm.relay.transform.MergeComposite(patterns)(mod)
    mod = tvm.relay.transform.AnnotateTarget('ilaflex')(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print('[Python] Transformation complete')
    mod = relay.transform.InferType()(mod)
    return mod

def test_compile(filename, relay_src, print_model=True, depth=1):
    relay_code = open(relay_src, 'r').read()
    with open(filename, 'r') as fp:
        recexpr_json = json.load(fp)
        compiler = RecExprCompiler({
            'flex-linear': 'ilaflex.linear'
        }, {
            'flex-linear': 'ilaflex'
        })

        original_model = tvm.parser.fromtext(relay_code)
        shape_dict = dict()
        for args in original_model['main'].params:
            shape_dict[args.name_hint] = tuple(args.type_annotation.shape)

        start = time.time()
        expr = compiler.to_relay_expr(recexpr_json, shape_dict)
        mod = tvm.ir.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        end = time.time()
        baseline = ResMLP(
            image_size = 32,
            patch_size = 16,
            dim = 64,
            depth = depth,
            num_classes = 32
        )
        if print_model:
            with open('resmlp-compiled.relay', 'w') as fp:
                fp.write(str(mod))
                print(f'[DEBUG] Time to compile from RecExpr: {(end - start) * 1000} ms')
                print(f'[INFO] Compiled model saved to resmlp-compiled.relay')
            return

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
        with tvm.transform.PassContext(opt_level=0):
            for target, dev in tvm.testing.enabled_targets():
                # print(f'target: {target} | device: {dev}')
                start = time.time()
                relay_graph, lib, params = relay.build(mod, target=target, params=n_params)
                end = time.time()
                relay_model = graph_executor.create(relay_graph, lib, dev)
                relay_model.set_input(**params)
                for name, inp in compiled_input.items():
                    relay_model.set_input(name, inp)
                relay_model.run()
                for i, baseline_output in enumerate(baseline_outputs):
                    compiled_output = relay_model.get_output(i).asnumpy()
                    tvm.testing.assert_allclose(compiled_output, baseline_outputs[i])
                    print(f'[DEBUG] Building time: {(end - start) * 1000} ms')
                    print('result:')
                    print(compiled_output)
                break

if __name__ == '__main__':
    import sys
    test_compile(sys.argv[1], sys.argv[2], depth=int(sys.argv[3]), print_model=int(sys.argv[4]) == 1)
