import tvm
from tvm import relay

def workload(batch, in_feat, hidden_0, hidden_1):
    data = relay.var('data', relay.TensorType((batch, in_feat)))
    weights_0 = relay.var('weights_0', relay.TensorType((hidden_0, in_feat)))
    weights_1 = relay.var('weights_1', relay.TensorType((hidden_1, hidden_0)))
    kernel = relay.var('kernel_0', relay.TensorType((3, 1, batch, hidden_1)))
    fc1 = relay.nn.dense(data, weights_0)
    act1 = relay.nn.relu(fc1)
    fc2 = relay.nn.dense(act1, weights_1)
    act2 = relay.nn.relu(fc2)
    mod = tvm.ir.IRModule.from_expr(act2)
    mod = relay.transform.InferType()(mod)
    with open('example_workload.relay', 'w') as fp:
        fp.write(mod.astext())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    parser.add_argument('in_feat', type=int)
    parser.add_argument('hidden_0', type=int)
    parser.add_argument('hidden_1', type=int)
    args = parser.parse_args()
    workload(args.batch, args.in_feat, args.hidden_0, args.hidden_1)
