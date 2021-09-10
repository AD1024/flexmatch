import json
import tvm
from tvm import relay
from megraph.language import RecExprCompiler

def test_compile(filename):
    with open(filename, 'r') as fp:
        recexpr_json = json.load(fp)
        compiler = RecExprCompiler({
            'flex-linear': 'ilaflex'
        }, {
            'flex-linear': 'ilaflex.linear'
        })
        # %input0: Tensor[(1, 3, 32, 32), float32]
        # %v1_weight: Tensor[(64, 768), float32]
        # %v1_bias: Tensor[(64), float32]
        # %v2_0_affine_g: Tensor[(1, 1, 64), float32]
        # %v2_0_affine_b: Tensor[(1, 1, 64), float32]
        # %v2_0_fn_weight: Tensor[(4, 4, 1), float32]
        # %v2_0_fn_bias: Tensor[(4), float32]
        # %v2_0_scale: Tensor[(1, 1, 64), float32]
        # %v2_1_affine_g: Tensor[(1, 1, 64), float32]
        # %v2_1_affine_b: Tensor[(1, 1, 64), float32]
        # %v2_1_fn_0_weight: Tensor[(256, 64), float32]
        # %v2_1_fn_0_bias: Tensor[(256), float32]
        # %v2_1_fn_2_weight: Tensor[(64, 256), float32]
        # %v2_1_fn_2_bias: Tensor[(64), float32]
        # %v2_1_scale: Tensor[(1, 1, 64), float32]
        # %v3_g: Tensor[(1, 1, 64), float32]
        # %v3_b: Tensor[(1, 1, 64), float32]
        # %v5_weight: Tensor[(32, 64), float32], %v5_bias: Tensor[(32), float32]
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
            'v3_g': (1, 1, 64),
            'v3_b': (1, 1, 64),
            'v5_weight': (32, 64),
            'v5_bias': (32,)
        })
        mod = tvm.ir.IRModule.from_expr(expr)
        print(mod)
        print(relay.transform.InferType()(mod))

if __name__ == '__main__':
    test_compile('tests/resmlp-dump.json')