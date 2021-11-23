import tvm
from tvm import relay

def workload(batch, in_feat, hidden_0):
    data = relay.var('data', relay.TensorType((batch, in_feat)))
    weights_0 = relay.var('weights_0', relay.TensorType((in_feat, in_feat)))
    weights_1 = relay.var('weights_1', relay.TensorType((in_feat, in_feat)))
    weights_2 = relay.var('weights_2', relay.TensorType((hidden_0, in_feat)))
    fc1 = relay.nn.dense(data, weights_0)
    act1 = relay.nn.relu(fc1)
    fc2 = relay.nn.dense(act1, weights_1)
    act2 = relay.nn.relu(fc2)
    fc3 = relay.nn.relu(act2)
    act3 = relay.nn.dense(fc3, act1)
    res = relay.nn.relu(act3)
    mod = tvm.ir.IRModule.from_expr(res)
    mod = relay.transform.InferType()(mod)
    with open('example_reuse_workload.relay', 'w') as fp:
        fp.write(mod.astext())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    parser.add_argument('in_feat', type=int)
    parser.add_argument('hidden_0', type=int)
    args = parser.parse_args()
    workload(args.batch, args.in_feat, args.hidden_0)
