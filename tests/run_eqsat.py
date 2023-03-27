import os
import subprocess
import argparse
import sys


def main(relay_file, output_filename, configs, use_ilp):
    home_dir = os.environ.get('FLEXMATCH_HOME')
    if home_dir is None:
        print('FLEXMATCH_HOME not set, skipping...')
        return
    cur_dir = os.getcwd()
    relay_file = os.path.join(cur_dir, relay_file)
    model_rewrite_file = os.path.join(
        cur_dir, f'{output_filename}-rewritten.json')
    analysis_data_file = os.path.join(cur_dir, f'{output_filename}-data.json')
    configs = list(map(lambda x: f'{x}.json', configs))
    for config in configs:
        if not os.path.isfile(os.path.join(home_dir, 'configs', config)):
            raise Exception(f'{config} is not a valid config json')
    cmd = './target/debug/flexmatch {} {} {} {}'.format(
        relay_file,
        os.path.join(cur_dir, f'{output_filename}-rewritten.json'),
        os.path.join(cur_dir, f'{output_filename}-data.json'),
        ' '.join(configs)
    )
    # print('GOT HEREE!!!!! -<<DF_AF<>', os.path.join(home_dir, 'flexmatch'))
    # print(['./target/debug/flexmatch',
    #        relay_file,
    #        model_rewrite_file,
    #        analysis_data_file])
    try:
        subprocess.run(cwd=os.path.join(home_dir, 'flexmatch'),
                       args=['./target/debug/flexmatch',
                             relay_file,
                             model_rewrite_file,
                             analysis_data_file]
                       + configs
                       + (['--ilp'] if use_ilp else []),
                       stdout=sys.stdout.buffer,
                       check=True)
    except subprocess.CalledProcessError as e:
        print('Error caught when running EqSat ({}):\n{}', e.returncode, str(e))
        return
    else:
        print('Output file written to: ',
              model_rewrite_file,
              analysis_data_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--relay-file', dest='relay_file',
                        type=str, help='Relay source file', required=True)
    parser.add_argument('--output-file', dest='output_file', type=str,
                        help='Output file name of extracted model and analysis data (they share the same name)', required=True)
    parser.add_argument('--configs', dest='configs', nargs='+',
                        help='Equality Saturation Configs', required=True)
    parser.add_argument('--use-ilp', dest='use_ilp', action='store_true')
    args = parser.parse_args()
    main(args.relay_file, args.output_file, args.configs, args.use_ilp)
