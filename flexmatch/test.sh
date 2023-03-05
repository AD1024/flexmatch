#!/bin/bash
models=(mobilenetv2.relay resmlp.relay resnet18.relay bert.relay efficientnet.relay resnet50_simplifyinference_from_pytorch.relay)

for relay_file in ${models[@]}; do
    cargo run --features cplex ../tests/models/$relay_file rewrite.json data.json vta-dense.json im2col.json --eval
done