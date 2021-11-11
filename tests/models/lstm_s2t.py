"""
LSTM definition taken from
https://github.com/uwsampl/3la-tvm/blob/3la-pldi-push-main/tests/python/byo3la/end_to_end_speech_to_text.py
"""
import tvm
from tvm import relay
import numpy as np

def relay_lstm_cell(batch_size, input_size, hidden_size):
    # based on https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
    state_tensor_type = relay.TensorType((batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    inp = relay.var("input", shape=(batch_size, input_size))
    state = relay.Var("state", type_annotation=state_tuple_type)

    w_ih = relay.var("w_ih", shape=(4*hidden_size, input_size))
    w_hh = relay.var("w_hh", shape=(4*hidden_size, hidden_size))
    b_ih = relay.var("b_ih", shape=(4*hidden_size,))
    b_hh = relay.var("b_hh", shape=(4*hidden_size,))

    hidden = relay.TupleGetItem(state, 0)
    cell_state = relay.TupleGetItem(state, 1)

    # PyTorch packs the i2h and h2h weights and biases together so we will match that here
    w_i_splits = relay.split(w_ih, 4, 0)
    w_h_splits = relay.split(w_hh, 4, 0)
    b_i_splits = relay.split(b_ih, 4, 0)
    b_h_splits = relay.split(b_hh, 4, 0)
    w_ii, w_if, w_ig, w_io = w_i_splits[0], w_i_splits[1], w_i_splits[2], w_i_splits[3]
    w_hi, w_hf, w_hg, w_ho = w_h_splits[0], w_h_splits[1], w_h_splits[2], w_h_splits[3]
    b_ii, b_if, b_ig, b_io = b_i_splits[0], b_i_splits[1], b_i_splits[2], b_i_splits[3]
    b_hi, b_hf, b_hg, b_ho = b_h_splits[0], b_h_splits[1], b_h_splits[2], b_h_splits[3]

    def weighted_value(weight, value, bias):
        return relay.transpose(relay.nn.dense(weight, value) + relay.reshape(bias, (hidden_size, 1)))

    i_t = relay.sigmoid(weighted_value(w_ii, inp, b_ii) + weighted_value(w_hi, hidden, b_hi))
    f_t = relay.sigmoid(weighted_value(w_if, inp, b_if) + weighted_value(w_hf, hidden, b_hf))
    g_t = relay.tanh(weighted_value(w_ig, inp, b_ig) + weighted_value(w_hg, hidden, b_hg))
    o_t = relay.sigmoid(weighted_value(w_io, inp, b_io) + weighted_value(w_ho, hidden, b_ho))
    c_t = f_t*cell_state + i_t*g_t
    h_t = o_t*relay.tanh(c_t)

    h_var = relay.Var("h")
    c_var = relay.Var("c")
    return relay.Function([inp, state, w_ih, w_hh, b_ih, b_hh],
                          relay.Let(h_var, h_t,
                                    relay.Let(c_var, c_t,
                                              relay.Tuple([h_var, relay.Tuple([h_var, c_var])]))),
                          ret_type=relay.TupleType([state_tensor_type, state_tuple_type]))


def lstm_body(data, state, i2h_weight, h2h_weight, i2h_bias, h2h_bias,
              batch_size, input_size, hidden_size, time_steps, time_axis=1):
    builder = relay.ScopeBuilder()
    cell = builder.let("lstm_cell", relay_lstm_cell(batch_size, input_size, hidden_size))
    splits = builder.let("splits", relay.split(data, time_steps, time_axis).astuple())
    last_state = state
    seq_outs = []
    for i in range(time_steps):
        squeezed = builder.let(f"squeezed_{i}", relay.squeeze(relay.TupleGetItem(splits, i), axis=[time_axis]))
        cell_out = builder.let(f"cell_out_{i}",
                               cell(squeezed, last_state,
                                    i2h_weight, h2h_weight,
                                    i2h_bias, i2h_bias))
        new_seq_out = builder.let(f"seq_out_{i}", relay.TupleGetItem(cell_out, 0))
        seq_outs.append(new_seq_out)
        new_hidden = builder.let(f"state_update_{i}", relay.TupleGetItem(cell_out, 1))
        last_state = new_hidden

    stacked = builder.let("stacked", relay.stack(seq_outs, axis=time_axis))
    # builder.ret(relay.Tuple([stacked, reshape_hidden, reshape_cell]))
    builder.ret(relay.Tuple([stacked]))
    return builder.get()


# Warning! This is an unrolled RNN! If you want a truly dynamic RNN,
# you should define it using a list ADT and apply the LSTM cell recursively.
# We can easily do that, though note that interacting
# with the ADT objects in the BYOC codegen would be tricky
def lstm_definition(batch_size, input_size, hidden_size, time_steps,
                    time_axis=1):
    """
    Wrap the LSTM body in a function
    """
    state_tensor_type = relay.TensorType((batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    input_var = relay.var("input", shape=(batch_size, time_steps, input_size))
    state_var = relay.var("state", type_annotation=state_tuple_type)
    i2h_weight_var = relay.var("i2h_weight", shape=(4*hidden_size, input_size))
    h2h_weight_var = relay.var("h2h_weight", shape=(4*hidden_size, hidden_size))
    i2h_bias_var = relay.var("i2h_bias", shape=(4*hidden_size,))
    h2h_bias_var = relay.var("h2h_bias", shape=(4*hidden_size,))

    ret_type = relay.TupleType([
        relay.TensorType((batch_size, time_steps, hidden_size)),
        relay.TensorType((1, batch_size, hidden_size)),
        relay.TensorType((1, batch_size, hidden_size))
    ])

    return relay.Function(
        [input_var, state_var, i2h_weight_var, h2h_weight_var,
         i2h_bias_var, h2h_bias_var],
        lstm_body(input_var, state_var,
                  i2h_weight_var, h2h_weight_var, i2h_bias_var, h2h_bias_var,
                  batch_size, input_size, hidden_size, time_steps, time_axis=time_axis),
        ret_type=ret_type)

def get_lstm_pattern(batch_size, input_size, hidden_size, time_steps):
    lstm_pattern = lstm_definition(batch_size, input_size, hidden_size, time_steps).body
    mod = tvm.ir.IRModule.from_expr(lstm_pattern)
    mod = relay.transform.SimplifyInference()(mod)
    with open('lstm_pattern.relay', 'w') as fp:
        fp.write(mod.astext())
        print('Pattern written to lstm_pattern.relay')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--input_size', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--time_steps', type=int, required=True)
    args = parser.parse_args()
    get_lstm_pattern(args.batch, args.input_size, args.hidden_size, args.time_steps)
