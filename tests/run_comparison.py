import tvm
import numpy as np
import tvm.testing
from tvm import relay
from tvm.ir import transform
from tvm.contrib import graph_executor

def run_file(src, **params):
    with open(src, 'r') as fp:
        relay_src = fp.read()
        mod = tvm.parser.fromtext(relay_src)
        print(mod)
        mod = relay.transform.InferType()(mod)
        with tvm.transform.PassContext(opt_level=0):
            # mod = relay.build()
            for target, dev in tvm.testing.enabled_targets():
                relay_graph, lib, params = relay.build(mod, target=target, params=params)
                relay_model = graph_executor.create(relay_graph, lib, dev)
                relay_model.set_input(**params)
                relay_model.run()
                return relay_model.get_output(0)

def main(lhs_src, rhs_src):
    inputs = np.random.rand(1, 3, 32).astype('float32')
    weights = np.random.rand(8, 3, 3).astype('float32')
    lhs_res = run_file(lhs_src, data=inputs, weights=weights)
    rhs_res = run_file(rhs_src, data=inputs, weights=weights)
    # tvm.testing.assert_allclose(lhs_res, rhs_res)
    tvm.testing.assert_allclose(lhs_res.asnumpy(), rhs_res.asnumpy())

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])