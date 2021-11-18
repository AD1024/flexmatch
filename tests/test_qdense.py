import numpy as np
import tvm
import math
from tvm import relay
import tvm.testing
import time

params = {
    'data': np.random.randn(2, 2).astype('float32'),
    'weights': np.random.randn(2, 2).astype('float32'),
}

def approximate(x):
    eps = 1e-7
    n = 1
    d = 1
    f = n / d
    while math.fabs(f - x) > eps:
        if f < x:
            n += 1
        else:
            d += 1
            n = round(x * d)
        f = n / d
    nbits_up = int(math.ceil(math.log(d) / math.log(2)))
    nbits_down = int(math.floor(math.log(d) / math.log(2)))
    print('Now::', n, d, nbits_up, nbits_down)
    round_up_d = 1 << nbits_up
    round_down_d = 1 << nbits_down
    nbits = nbits_up
    if round_up_d - d > d - round_down_d:
        round_up_d = round_down_d
        nbits = nbits_down
    fact = round_up_d / d
    n *= fact
    n = round(n)
    return n, nbits

def quantize_uniform(x, num_bits=8):
    qmin = -2.**(num_bits-1) + 1
    qmax = 2.**(num_bits-1) - 1
    scale_p = x.max() / qmax
    scale_n = x.min() / qmin
    scale = max(scale_p, scale_n)
    #print ("integer scaling factor is: ", scale)
    q_x = (x / scale).round()
    #print ("integer-quantized weights: ", q_x)

    return q_x, scale

def run_file(src, params):
    print(f'Compiling & Running: {src}')
    with open(src, 'r') as fp:
        relay_src = fp.read()
        start = time.time()
        mod = tvm.parser.fromtext(relay_src)
        mod = relay.transform.InferType()(mod)
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
            qd, S_data = quantize_uniform(inputs[0])
            qw, S_w = quantize_uniform(inputs[1])
            _, S_act = quantize_uniform(np.matmul(inputs[0], np.transpose(inputs[1])))
            qd = qd.astype('int32')
            qw = qw.astype('int32')
            qact = np.matmul(qd, np.transpose(qw))
            factor, nbits = approximate(float(S_data * S_w / S_act))
            acc_buf = ((qact * factor) >> nbits)
            acc_buf = np.minimum(np.maximum(acc_buf, -127), 127)
            print(acc_buf)
            output_buf = (acc_buf & 0b11111111).astype('int8')
            print('Scale: ', S_data * S_w / S_act)
            print('Approximate', approximate(S_data * S_w / S_act))
            print('Approx Result:', output_buf.astype('float32') * S_act)
            print()
            return result
import numpy as np
print(params)
print(run_file('qdense.relay', params))
print(np.matmul(params['data'], np.transpose(params['weights'])))
