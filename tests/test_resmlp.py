# Res-MLP implementation taken from https://github.com/lucidrains/res-mlp-pytorch/blob/7a5b5276cd9270ad8131f77dfe4e6f56fe65fb3f/res_mlp_pytorch/res_mlp_pytorch.py
from megraph.matcher import check_and_annotate
import torch
import tvm
import megraph
from tvm import relay
import tvm.testing
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import ilavta
from tvm.relay.op.contrib import ilaflex
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from tvm.relay import ExprMutator


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):  # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


def ResMLP(*, image_size, patch_size, dim, depth, num_classes, expansion_factor=4):
    assert (image_size % patch_size) == 0, "image must be divisible by patch size"
    num_patches = (image_size // patch_size) ** 2
    wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

    return nn.Sequential(
        Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        ),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[
            nn.Sequential(
                wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
                wrapper(
                    i,
                    nn.Sequential(
                        nn.Linear(dim, dim * expansion_factor),
                        nn.GELU(),
                        nn.Linear(dim * expansion_factor, dim),
                    ),
                ),
            )
            for i in range(depth)
        ],
        Affine(dim),
        Reduce("b n c -> b c", "mean"),
        nn.Linear(dim, num_classes)
    )

def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


class RenameMutator(ExprMutator):
    def __init__(self):
        super().__init__()
        self.var_map = dict()
    
    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        else:
            if '.' in var.name_hint:
                new_name = var.name_hint.replace('.', '_')
                new_var = relay.Var(new_name, type_annotation=var.type_annotation)
                self.var_map[var] = new_var
                return new_var
            else:
                return var

def verify_model(
    model_name,
    input_data=[],
    custom_convert_map={},
    rtol=1e-5,
    atol=1e-5,
    expected_ops=[],
    print_model=True,
    run_comparison=False,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    if isinstance(model_name, str):
        baseline_model, baseline_input = load_model(model_name)
    elif isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or len(input_data.shape) == 0:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"

    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    trace = torch.jit.trace(baseline_model, [input.clone() for input in baseline_input])
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()

        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()

    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    compiled_input = dict(
        zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input])
    )
    egraph = megraph.load_egraph('tests/linear_pattern.egraph')
    # Dump the model with renamed variables (seems rust frontend cannot parse variables with dots in their names)
    mutator = RenameMutator()
    new_body = mutator.visit(mod["main"].body)
    new_mod = tvm.ir.IRModule.from_expr(relay.Function(relay.analysis.free_vars(new_body), new_body))
    new_mod = relay.transform.InferType()(new_mod)
    with open('resmlp.relay', 'w') as fp:
        fp.write(new_mod.astext())
    return
    mut_main = check_and_annotate(mod["main"].body, (egraph, 'ilaflex', 'ilaflex.linear'))
    mod = tvm.ir.IRModule.from_expr(mut_main)
    if print_model:
        print(tvm.relay.transform.InferType()(mod))
    # return
    if not run_comparison:
        return

    with tvm.transform.PassContext(opt_level=3):
        for target, dev in tvm.testing.enabled_targets():
            relay_graph, relay_lib, relay_params = relay.build(
                mod, target=target, params=params
            )
            relay_model = graph_executor.create(relay_graph, relay_lib, dev)
            relay_model.set_input(**relay_params)
            for name, inp in compiled_input.items():
                relay_model.set_input(name, inp)
            relay_model.run()

            for i, baseline_output in enumerate(baseline_outputs):
                compiled_output = relay_model.get_output(i).asnumpy()

                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(
                    baseline_output, compiled_output, rtol=rtol, atol=atol
                )

    del model_name
    del baseline_model
    torch.cuda.empty_cache()

def main():
    model = ResMLP(
        image_size = 32,
        patch_size = 16,
        dim = 64,
        depth = 3,
        num_classes = 32
    )
    img = torch.randn(1, 3, 32, 32)
    verify_model(
        model.eval(),
        input_data=[img],
        print_model=True,
        run_comparison=True,
    )


if __name__ == "__main__":
    main()