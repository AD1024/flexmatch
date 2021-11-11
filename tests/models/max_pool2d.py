import tvm
import argparse
from tvm import relay

def main(b, c, h, w, kh, kw):
    data = relay.var('data', type_annotation=relay.TensorType((b, c, h, w)))
    result = relay.nn.max_pool2d(data, [kh, kw])
    mod = tvm.ir.IRModule.from_expr(result)
    mod = relay.transform.InferType()(mod)
    with open('max_pool2d.relay', 'w') as fp:
        fp.write(mod.astext())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a single maxpool 2d operator')
    parser.add_argument('b', type=int, help='Number of batches')
    parser.add_argument('c', type=int, help='Number of channels')
    parser.add_argument('h', type=int, help='height')
    parser.add_argument('w', type=int, help='width')
    parser.add_argument('kh', type=int, help='window height')
    parser.add_argument('kw', type=int, help='window width')
    
    args = parser.parse_args()
    main(args.b, args.c, args.h, args.w, args.kh, args.kw)
