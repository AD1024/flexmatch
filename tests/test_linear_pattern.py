import megraph
import tvm
from tvm import relay
from tvm.relay import Expr
from megraph.egraph_constructor import Constructor
from megraph.language import downcast
from megraph.matcher import check_and_annotate

def relay_linear(batch: int, in_channels: int, hidden_dim: int) -> Expr:
    t_x = relay.TensorType((batch, in_channels))
    t_w = relay.TensorType((hidden_dim, in_channels))
    t_b = relay.TensorType((hidden_dim,))
    x = relay.var('x', t_x)
    w = relay.var('w', t_w)
    b = relay.var('b', t_b)

    fc = relay.nn.dense(x, w)
    reshape_fc = relay.reshape(fc, (1, batch, hidden_dim))
    return relay.Function([x, w, b], relay.add(reshape_fc, b))

def relay_linear_original(batch: int, in_channels: int,  hidden_dim: int) -> Expr:
    t_x = relay.TensorType((batch, in_channels))
    t_w = relay.TensorType((hidden_dim, in_channels))
    t_b = relay.TensorType((hidden_dim,))
    x = relay.var('x', t_x)
    w = relay.var('w', t_w)
    b = relay.var('b', t_b)

    return relay.Function([x, w, b], relay.reshape(relay.nn.bias_add(relay.nn.dense(x, w), b), (1, batch, hidden_dim)))

def relay_linear_with_subexpr(batch: int, in_channels: int,  hidden_dim: int) -> Expr:
    t_x = relay.TensorType((batch, in_channels))
    t_y = relay.TensorType((batch, in_channels))
    t_w = relay.TensorType((hidden_dim, in_channels))
    t_b = relay.TensorType((hidden_dim,))
    x = relay.var('data', t_x)
    y = relay.var('y', t_y)
    w = relay.var('weight', t_w)
    b = relay.var('bias', t_b)

    return relay.Function([x, y, w, b],
            relay.reshape(relay.nn.bias_add(relay.nn.dense(relay.add(x, y), w), b), (1, batch, hidden_dim)))

def main():
    egraph = megraph.load_egraph('tests/linear_pattern.egraph')
    print(check_and_annotate(relay_linear(1, 32, 32).body, (egraph, 'ilaflex', 'ilaflex.linear')))
    print(check_and_annotate(relay_linear_original(1, 32, 32).body, (egraph, 'ilaflex', 'ilaflex.linear')))
    print(check_and_annotate(relay_linear_with_subexpr(1, 32, 32).body, (egraph, 'ilaflex', 'ilaflex.linear')))

if __name__ == '__main__':
    main()