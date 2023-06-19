import subprocess
import sys
import argparse
import numpy
import tvm
import test_model_on_vta as utils 
# from tvm.relay.testing import op_summary

DEFAULT_CONFIGS = {
    'max_pool2d': ['flexasr-maxpool'],
    'resnet18': ['hlscnn-conv2d'],
    'mobilenet': ['hlscnn-conv2d', 'linear-rewrites'],
    'mobilenetv2': ['im2col', 'vta-dense'],
    'resmlp': ['linear-rewrites'],
    'efficientnet': ['hlscnn-conv2d']
}

def run_eqsat(model, configs, use_ilp):
    try:
        subprocess.run(['python3',
                        'run_eqsat.py',
                        '--relay-file',
                        f'./models/{model}.relay',
                        f'--output-file',
                        model,
                        '--configs'] + configs + (['--use-ilp'] if use_ilp else []),
                        stdout=sys.stdout.buffer,
                         check=True)
        return (f'{model}-rewritten.json', f'{model}-data.json')
    except subprocess.CalledProcessError as e:
        print(f'Error while running eqsat:\n {e}')
        exit(1)

def compile_model(model, model_json, data_json, configs, debug):
    try:
        subprocess.run(['python3', 'compile_model.py',
                    f'./models/{model}.relay',
                    f'{model}-rewritten.relay', 
                    model_json,
                    data_json] + configs + (['--debug'] if debug else []),
                    stdout=sys.stdout.buffer,
                    stderr=sys.stderr.buffer,
                    check=True)
        return f'{model}-rewritten.relay'
    except subprocess.CalledProcessError as e:
        print(f'Error while compiling:\n {e}')
        exit(1)

def run_comparison(model, compiled):
    try:
        subprocess.run(['python3', 'run_comparison.py',
                        f'./models/{model}.relay',
                        compiled], stdout=sys.stdout.buffer, stderr=sys.stderr.buffer, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error while running comparison:\n {e}')
        exit(1)
    print('Passed!')

def quantize_model(mod):
    params = {
        k: numpy.random.randn(*v) / 1000.0
        for (k, v) in map(lambda x: (x.name_hint, x.type_annotation.shape), mod['main'].params[1:])
    }
    return utils.run_with_relay_quantization(mod, params, run=False)

def get_offload_stats(model, quantize):
    with open(model) as fp:
        src = fp.read()
        mod = tvm.parser.fromtext(src)
        if quantize:
            mod = quantize_model(mod)
        # print(f'ALL overloads: {op_summary.count_all_overloads(mod)}')
        # print(f'ALL Ops: {op_summary.count_all_ops(mod)}')
        # print(f'ALL Ops in overloads = {op_summary.count_all_ops_in_overloads(mod)} * #ops per pattern')

def run_model(model, configs, use_ilp, debug, get_stats, quantize):
    print(f'Step 1: Run EqSat with {" ".join(configs)}')
    (model_json, data_json) = run_eqsat(model, configs, use_ilp)
    print(f'Step 2: Compiling back to Relay with {model_json} and {data_json}')
    compiled = compile_model(model, model_json, data_json, configs, debug)
    if get_stats:
        print('Vanilla relay:')
        get_offload_stats(f'./models/{model}.relay', quantize)
        print('EqSat model:')
        get_offload_stats(compiled, quantize)
    else:
        print('Step 3: Running numeric comparisons')
        run_comparison(model, compiled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model to run')
    parser.add_argument('--configs', nargs='+', dest='configs', required=False, help='configs pass to eqsat')
    parser.add_argument('--default', required=False, dest='use_default', action='store_true', help='use pre-set config')
    parser.add_argument('--use-ilp', required=False, dest='use_ilp', action='store_true', help='use ILP extraction')
    parser.add_argument('--debug', required=False, dest='debug', action='store_true', help='use debug function to substitute accelerator calls')
    parser.add_argument('--get-stats', required=False, dest='get_stats', action='store_true')
    parser.add_argument('--quantize', required=False, dest='quantize', action='store_true')
    args = parser.parse_args()
    if args.use_default:
        if args.configs:
            print('Warning: overwriting configs with defaults')
        args.configs = DEFAULT_CONFIGS[args.model]
    if not args.configs:
        print('No config specified, skipping...')
        exit(0)
    run_model(args.model,
              args.configs if not args.use_default else DEFAULT_CONFIGS[args.model],
              args.use_ilp,
              args.debug,
              args.get_stats,
              args.quantize)
