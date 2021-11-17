import logging
import os
import tqdm
import tvm
import numpy as np
import torch
from tvm import relay
from tvm.contrib import graph_executor
from tvm.runtime.ndarray import cpu
from models.mobilenetv2 import MobileNetV2
import torchvision
import torchvision.transforms as transforms
import models
import test_vta_quantization as quant_utils

# Data prep code taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# trainset = torchvision.datasets.CIFAR10('./data', download=True, transform=transform_train)
# trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def get_relay_model(param_file, input_shape=(1, 3, 32, 32)):
    params = torch.load(param_file)
    prefix = "module."
    params = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in params.items()}
    model = MobileNetV2()
    model.load_state_dict(params)
    trace = torch.jit.trace(model, torch.randn(*input_shape))
    inputs = [('input0', input_shape)]
    return relay.frontend.from_pytorch(trace, inputs)

def test_relay_model(mod, params):
    print(mod)
    with tvm.transform.PassContext(opt_level=3):
        relay_graph, lib, params = relay.build(mod, params=params, target='llvm')
        graph_rt = graph_executor.create(relay_graph, lib, device=cpu(0))
        graph_rt.set_input(**params)
        total = 0
        correct = 0
        for idx, (inp, targets) in enumerate(testloader):
            graph_rt.set_input('input0', inp.numpy().astype('float32'))
            graph_rt.run()
            output = graph_rt.get_output(0).asnumpy()
            prediected = np.argmax(output, axis=1)
            total += targets.size(0)
            correct += np.sum(np.equal(prediected, targets.numpy()))
            if idx % 100 == 0:
                print(f'Batch #{idx}, Accuracy: {correct / total}')
            if idx > 2000:
                break
        print(f'Final accuracy: {correct / total}')

def get_cali_data():
    logging.info('Calibration:')
    total = len(testloader)
    for (inp, _) in tqdm.tqdm(testloader, total=total):
        yield {'input0': inp.cpu().numpy()}

def bind_params(func, params):
    """Bind the params to the expression."""
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.expr.const(v)
    return relay.expr.bind(func, bind_dict)

def run_with_relay_quantization(mod, params, run=True):
    BASE_CFG = {
        "skip_conv_layers": [],
        "skip_dense_layers": False,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
    }
    mod['main'] = models.utils.LetInliner().visit(mod['main'])
    mod['main'] = bind_params(mod['main'], params)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod['main'] = models.utils.AlterDense().visit(mod['main'])
    with relay.quantize.qconfig(**BASE_CFG, weight_scale='power2', calibration_mode='kl_divergence', skip_dense_layer=False):
        qmod = relay.quantize.quantize(mod, params=params, dataset=list(get_cali_data()))
    if run:
        test_relay_model(qmod, None)
    return qmod

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-model', required=False, dest='save_model', action='store_true')
    parser.add_argument('--relay-model', required=False, dest='relay_model')
    parser.add_argument('--quantize', required=False, dest='quantize', action='store_true')
    parser.add_argument('--layerwise', required=False, dest='layerwise_debug', action='store_true')
    parser.add_argument('--params', required=True, dest='params')
    args = parser.parse_args()
    # param_file = 'params/final_mobilenet_cifar10_400_epochs.pth'
    param_file = args.params
    if args.save_model:
        mod, params = get_relay_model(param_file)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.SimplifyInference()(mod)
        mod['main'] = models.RenameMutator({'.': '_'}).visit(mod['main'])
        mod = relay.transform.InferType()(mod)
        with open(os.path.join('./models/mobilenetv2.relay'), 'w') as fp:
            fp.write(mod.astext())
        exit(0)
    else:
        params = params = torch.load(param_file, map_location=torch.device('cpu'))
        prefix = "module."
        params = {(k[len(prefix):].replace('.', '_') if k.startswith(prefix) else k): v.cpu().numpy() for k, v in params.items()}
        if not args.relay_model:
            raise Exception('relay model not set')
        with open(args.relay_model, 'r') as fp:
            relay_src = fp.read()
            mod = tvm.parser.fromtext(relay_src)
            if args.layerwise_debug:
                inp = [{'input0': next(enumerate(testloader))[1][0], **params}]
                quant_utils.lockstep_layerwise(mod,  args.relay_model, inp)
            elif args.quantize:
                # run_with_relay_quantization(mod, params)
                # cali_dataset = get_cali_data()
                # calibrations = quant_utils.calibrate(mod, params, cali_dataset, ['nn.dense'])
                # mod = tvm.parser.fromtext(relay_src)
                expr = quant_utils.VTAQuantize([], ['nn.dense']).visit(mod['main'].body)
                mod = tvm.ir.IRModule.from_expr(expr)
                mod = relay.transform.InferType()(mod)
            test_relay_model(mod, params)
