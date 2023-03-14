#!/bin/bash
models=(mobilenetv2.relay resmlp.relay resnet18.relay efficientnet.relay resnet50_simplifyinference_from_pytorch.relay)
cargo build --features cplex
echo "Run EqSat + Extraction with Im2Col alone"
for i in {1..5}; do
    for relay_file in ${models[@]}; do
        ./target/debug/flexmatch ../tests/models/$relay_file rewrite.json data.json im2col.json --eval
    done
done

mv extraction_stats.txt extraction_stats_im2col.txt