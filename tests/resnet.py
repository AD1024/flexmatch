import tvm
import sys
import tvm.relay
import tvm.relay.testing

def main(batch, num_classes, num_layers, image_shape=(3, 32, 32)):
    model = tvm.relay.testing.resnet.get_net(batch, num_classes, num_layers, image_shape=image_shape)
    mod = tvm.ir.IRModule.from_expr(model.body)
    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.SimplifyInference()(mod)
    with open('resnet.relay', 'w') as fp:
        fp.write(mod.astext())
    
if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))