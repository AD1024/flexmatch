import tvm
import numpy as np
import tvm.testing
import time
from tvm import relay
from tvm.ir import transform
from tvm.contrib import graph_executor
from tvm.relay import ExprMutator

def get_inputs(src):
    with open(src, 'r') as fp:
        relay_src = fp.read()
        mod = tvm.parser.fromtext(relay_src)
        mod = relay.transform.InferType()(mod)
        inputs = dict()
        # inputs = []
        for var in mod['main'].params:
            shape = var.type_annotation.shape
            name_hint = var.name_hint
            inputs[name_hint] = np.random.rand(*[int(x) for x in shape]).astype('float32') / 100000.0
            # inputs.append(np.random.rand(*[int(x) for x in shape]).astype('float32'))
        return inputs

class LetInliner(ExprMutator):
    def __init__(self):
        super().__init__()
        self.let_map = {}

    def visit_var(self, var):
        if var in self.let_map:
            return self.let_map[var]
        return var

    def visit_let(self, let):
        # don't deal with functions
        if isinstance(let.value, relay.Function):
            return super().visit_let(let)
        self.let_map[let.var] = self.visit(let.value)
        return self.visit(let.body)

def run_file(src, params):
    print(f'Compiling & Running: {src}')
    with open(src, 'r') as fp:
        relay_src = fp.read()
        start = time.time()
        mod = tvm.parser.fromtext(relay_src)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.SimplifyInference()(mod)
        mod = tvm.ir.IRModule.from_expr(LetInliner().visit(mod["main"]))
        inputs = [params[x.name_hint] for x in mod['main'].params]
        for target, dev in tvm.testing.enabled_targets():
            # relay_graph, lib, params = relay.build(mod, target=target, params=params)
            executor = relay.create_executor('vm', mod=mod, device=dev, target=target).evaluate()
            end = time.time()
            print(f'compile time: {end - start}')
            # relay_model = graph_executor.create(relay_graph, lib, dev)
            # relay_model.set_input(**params)
            start = time.time()
            # relay_model.run()
            result = executor(*inputs)
            end = time.time()
            print(f'run time: {end - start}')
            # return relay_model.get_output(0)
            return result

def main(lhs_src, rhs_src):
    inputs = get_inputs(lhs_src)
    lhs_res = run_file(lhs_src, inputs)
    rhs_res = run_file(rhs_src, inputs)
    print(lhs_res)
    print(rhs_res)
    tvm.testing.assert_allclose(lhs_res.asnumpy(), rhs_res.asnumpy())

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
