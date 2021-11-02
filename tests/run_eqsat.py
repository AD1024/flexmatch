import os
import json
import sys


def main(relay_file, output_filename, *configs):
    home_dir = os.environ.get('FLEXMATCH_HOME')
    cur_dir = os.getcwd()
    relay_file = os.path.join(cur_dir, relay_file)
    if home_dir:
        rewrites = set()
        for config in configs:
            try:
                with open(os.path.join(home_dir, 'configs', f'{config}.json'), 'r') as fp:
                    rws = json.load(fp)
                    rewrites = rewrites.union(set(rws['rewrites']))
            except Exception as e:
                print(f'Exception caught when reading {config}:\n{e}')

        os.chdir(os.path.join(home_dir, 'flexmatch'))
        # os.system('cargo build')
        cmd = './target/debug/flexmatch {} {} {} {}'.format(
            relay_file,
            os.path.join(cur_dir, f'{output_filename}-rewritten.json'),
            os.path.join(cur_dir, f'{output_filename}-data.json'),
            ' '.join(list(rewrites))
        )
        os.system(cmd)
        print('Output file written to: ',
                os.path.join(cur_dir, f'{output_filename}-rewritten.json'),
                os.path.join(cur_dir, f'{output_filename}-data.json'))
    else:
        print('FLEXMATCH_HOME not set, skipping...')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'run_eqsat.py relay_src output_file configs+')
        exit(0)
    main(*sys.argv[1:])
