import megraph
import os
import json
import sys
import tvm
from tvm import relay
from tvm.relay import nn

def main(relay_file, output_filename, model_json, data_json, *configs, debug=False):
    home_dir = os.environ.get('FLEXMATCH_HOME')
    if home_dir:
        compilers = dict()
        composites = dict()
        debug_funcs = dict()
        for config in configs:
            try:
                with open(os.path.join(home_dir, 'configs', f'{config}.json'), 'r') as fp:
                    cfg = json.load(fp)
                    compilers.update(cfg.get('compilers', {}))
                    composites.update(cfg.get('composites', {}))
                    debug_funcs.update(
                        dict(map(lambda pi: (pi[0], eval(pi[1])),
                        cfg.get('debug_functions', {}).items()))
                    )
            except Exception as e:
                print(f'Error caught when reading {config}:\n{e}')
        with open(relay_file, 'r') as fp:
            src = fp.read()
            source_model = tvm.parser.fromtext(src)
        
        with open(model_json, 'r') as fp:
            expr_data = json.load(fp)
        with open(data_json, 'r') as fp:
            analysis_data = json.load(fp)
            analysis_data = dict(map(lambda pi: (int(pi[0]), pi[1]), analysis_data.items()))
        shape_dict = dict()
        for args in source_model['main'].params:
            shape_dict[args.name_hint] = tuple(args.type_annotation.shape)
        
        recexpr_compiler = megraph.RecExprCompiler(composites, compilers, debug_funcs)
        compiled_expr = recexpr_compiler.to_relay_expr(expr_data, shape_dict, analysis_data, use_debug_func=debug)
        mod = tvm.ir.IRModule.from_expr(compiled_expr)
        mod = relay.transform.InferType()(mod)
        with open(output_filename, 'w') as fp:
            fp.write(mod.astext())

        print(f'Compiled model saved to {output_filename}')
    else:
        print('FLEXMATCH_HOME not set')

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('compile_model.py relay_src output_file rewritten_json data_json configs Optional[debug]')
    else:
        main(*filter(lambda x: not x.startswith('--'), sys.argv[1:]), debug='--debug' in sys.argv)