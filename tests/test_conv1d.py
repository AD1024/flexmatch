import megraph
import tvm
import json
import sys
from tvm import relay

def main(json_file):
    name_to_shape = {
        'data': (1, 3, 32),
        'weights': (8, 3, 3)
    }
    with open(json_file, 'r') as fp:
        rec_expr = json.load(fp)
        expr = megraph.RecExprCompiler({
            'vta-dense': 'ilavta.dense'
        }, {
            'vta-dense': 'ilavta'
        }).to_relay_expr(rec_expr, name_to_shape)
        mod = tvm.ir.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        print(mod)

if __name__ == '__main__':
    main(sys.argv[1])
