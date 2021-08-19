import tvm
from tvm import relay
from tvm.relay import Expr
from megraph.egraph_constructor import Constructor
from megraph.language import downcast
from megraph.matcher import check_and_annotate

egraph_record = '''
SIZE 10
ROOT 10
ECLASS 0
BEGIN_ENODE
BEGIN_SYMBOL x END_SYMBOL
END_ENODE
END_ECLASS
ECLASS 10
BEGIN_ENODE
BEGIN_SYMBOL add END_SYMBOL
BEGIN_CHILDREN 6 7 
END_CHILDREN
END_ENODE
BEGIN_ENODE
BEGIN_SYMBOL reshape END_SYMBOL
BEGIN_CHILDREN 9 5 
END_CHILDREN
END_ENODE
END_ECLASS
ECLASS 7
BEGIN_ENODE
BEGIN_SYMBOL b END_SYMBOL
END_ENODE
END_ECLASS
ECLASS 4
BEGIN_ENODE
BEGIN_SYMBOL 4 END_SYMBOL
END_ENODE
END_ECLASS
ECLASS 1
BEGIN_ENODE
BEGIN_SYMBOL w END_SYMBOL
END_ENODE
END_ECLASS
ECLASS 5
BEGIN_ENODE
BEGIN_SYMBOL shape END_SYMBOL
BEGIN_CHILDREN 3 4 4 
END_CHILDREN
END_ENODE
END_ECLASS
ECLASS 2
BEGIN_ENODE
BEGIN_SYMBOL dense END_SYMBOL
BEGIN_CHILDREN 0 1 
END_CHILDREN
END_ENODE
END_ECLASS
ECLASS 9
BEGIN_ENODE
BEGIN_SYMBOL bias_add END_SYMBOL
BEGIN_CHILDREN 2 7 
END_CHILDREN
END_ENODE
BEGIN_ENODE
BEGIN_SYMBOL flex-linear END_SYMBOL
BEGIN_CHILDREN 0 1 7 
END_CHILDREN
END_ENODE
END_ECLASS
ECLASS 6
BEGIN_ENODE
BEGIN_SYMBOL reshape END_SYMBOL
BEGIN_CHILDREN 2 5 
END_CHILDREN
END_ENODE
END_ECLASS
ECLASS 3
BEGIN_ENODE
BEGIN_SYMBOL 1 END_SYMBOL
END_ENODE
END_ECLASS

FIN
'''

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

egraph = Constructor().from_text(egraph_record)
for i, eclass in egraph.eclasses.items():
    eclass.map(lambda x: downcast(x))

print(check_and_annotate(relay_linear(1, 32, 32).body, (egraph, 'ilaflex', 'ilaflex.linear')))
print(check_and_annotate(relay_linear_original(1, 32, 32).body, (egraph, 'ilaflex', 'ilaflex.linear')))
print(check_and_annotate(relay_linear_with_subexpr(1, 32, 32).body, (egraph, 'ilaflex', 'ilaflex.linear')))