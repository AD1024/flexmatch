import tvm
import sys
import tvm.relay
import tvm.relay.testing

def main(batch, num_classes, image_shape=(3, 32, 32)):
    mod, _ = tvm.relay.testing.mobilenet.get_workload(batch, num_classes, image_shape=image_shape)
    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.SimplifyInference()(mod)
    with open('mobilenet.relay', 'w') as fp:
        fp.write(mod.astext())
    
if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))