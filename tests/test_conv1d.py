import megraph
import tvm
import json
import sys
from tvm import relay

def main(json_file, analysis_file, relay_src):
    # name_to_shape = {
    #     'data': (1, 3, 32),
    #     'weights': (8, 3, 3)
    # }
    name_to_shape = megraph.load_weights(relay_src)
    with open(json_file, 'r') as fp:
        with open(analysis_file, 'r') as analysis_fp:
            eclass_analysis = json.load(analysis_fp)
            eclass_analysis = dict(map(lambda pi: (int(pi[0]), pi[1]), eclass_analysis.items()))
            rec_expr = json.load(fp)
            expr = megraph.RecExprCompiler({
                'vta-dense': 'ilavta.dense'
            }, {
                'vta-dense': 'ilavta'
            }).to_relay_expr(rec_expr, name_to_shape, analysis_data=eclass_analysis)
            mod = tvm.ir.IRModule.from_expr(expr)
            mod = relay.transform.InferType()(mod)
            # with open('conv1d-im2col.relay', 'w') as out_file:
            #     out_file.write(str(mod))

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'test_conv1d.py EXPR_JSON ANALYSIS_JSON RELAY_SRC')
        exit(1)
    main(sys.argv[1], sys.argv[2],sys.argv[3])
