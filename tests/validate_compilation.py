import subprocess
import sys
import argparse

DEFAULT_CONFIGS = {
    'max_pool2d': ['flexasr-maxpool'],
    'resnet18': ['hlscnn-conv2d'],
    'mobilenet': ['hlscnn-conv2d', 'linear-rewrites'],
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

def run_model(model, configs, use_ilp, debug):
    (model_json, data_json) = run_eqsat(model, configs, use_ilp)
    compiled = compile_model(model, model_json, data_json, configs, debug)
    run_comparison(model, compiled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model to run')
    parser.add_argument('--configs', nargs='+', dest='configs', required=False, help='configs pass to eqsat')
    parser.add_argument('--default', required=False, dest='use_default', action='store_true', help='use pre-set config')
    parser.add_argument('--use-ilp', required=False, dest='use_ilp', action='store_true', help='use ILP extraction')
    parser.add_argument('--debug', required=False, dest='debug', action='store_true', help='use debug function to substitute accelerator calls')
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
              args.debug)
