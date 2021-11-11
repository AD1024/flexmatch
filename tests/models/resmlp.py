# Res-MLP implementation taken from https://github.com/lucidrains/res-mlp-pytorch/blob/7a5b5276cd9270ad8131f77dfe4e6f56fe65fb3f/res_mlp_pytorch/res_mlp_pytorch.py
import torch
import argparse
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

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

def main(depth):
    import os
    import tvm
    from tvm import relay
    from tvm.relay import ExprMutator
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
    model = ResMLP(
                image_size = 32,
                patch_size = 16,
                dim = 64,
                depth = 12,
                num_classes = 32)
    inputs = [torch.randn(1, 3, 32, 32)]
    input_names = ["input{}".format(idx) for idx, inp in enumerate(inputs)]
    input_shapes = list(zip(input_names, [inp.shape for inp in inputs]))
    trace = torch.jit.trace(model, [input.clone() for input in inputs])
    mod, _ = relay.frontend.from_pytorch(trace, input_shapes, {})
    mod['main'] = RenameMutator().visit(mod['main'])
    with open(os.path.join(os.environ['FLEXMATCH_HOME'],
            'tests', 'models', f'resmlp-depth-{depth}.relay'), 'w') as fp:
        fp.write(mod.astext())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', dest='depth', type=int, required=True, help='Depth of ResMLP model')
    args = parser.parse_args()
    main(args.depth)