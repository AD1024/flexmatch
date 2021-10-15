import tvm
from tvm import relay
from tvm.relay.op.contrib import ilaflex

def run_passes(mod):
    patterns = ilaflex.pattern_table()
    mod = tvm.relay.transform.MergeComposite(patterns)(mod)
    mod = tvm.relay.transform.AnnotateTarget('ilaflex')(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print('[Python] Transformation complete')
    mod = relay.transform.InferType()(mod)
    return mod

def main(relay_src):
    with open(relay_src, 'r') as fp:
        relay_mod = tvm.parser.fromtext(fp.read())
        print('[Python] Running BYOC Pattern Matcher')
        relay_mod = run_passes(relay_mod)
        with open('resmlp-byoc.relay', 'w') as f:
            f.write(str(relay_mod))
            print('[Python] Transformed model rewritten to resmlp-byoc.relay')

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
