import os
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
    total = len(testloader)
    for idx, (inp, _) in enumerate(testloader):
        if idx > total / 4:
            break
        if idx % 100 == 0:
            print(f'Calibration WIP: {idx}')
        yield {'input0': inp.cpu().numpy()}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-model', required=False, dest='save_model', action='store_true')
    parser.add_argument('--relay-model', required=False, dest='relay_model')
    parser.add_argument('--quantize', required=False, dest='quantize', action='store_true')
    args = parser.parse_args()
    param_file = 'params/final_mobilenet_cifar10_400_epochs.pth'
    # test_relay_model(mod, params)
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
        params = params = torch.load(param_file)
        prefix = "module."
        params = {(k[len(prefix):].replace('.', '_') if k.startswith(prefix) else k): v.cpu().numpy() for k, v in params.items()}
        if not args.relay_model:
            raise Exception('relay model not set')
        with open(args.relay_model, 'r') as fp:
            mod = tvm.parser.fromtext(fp.read())
            if args.quantize:
                cali_dataset = get_cali_data()
                calibrations = quant_utils.calibrate(mod, params, cali_dataset, ['nn.dense'])
                expr = quant_utils.VTAQuantize(calibrations, ['nn.dense']).visit(mod['main'].body)
                mod = tvm.ir.IRModule.from_expr(expr)
            test_relay_model(mod, params)