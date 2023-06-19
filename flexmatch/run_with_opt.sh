#!/bin/bash
models=(mobilenetv2.relay resmlp.relay resnet18.relay efficientnet.relay resnet50_simplifyinference_from_pytorch.relay)
cargo build --features cplex
echo "Run EqSat + Extraction with Im2Col + Simplifications"
for i in {1..5}; do
    for relay_file in ${models[@]}; do
        echo ">>> Extract with WPMAXSAT"
        timeout 20 ./target/debug/flexmatch ../tests/models/$relay_file rewrite.json data.json simpl.json im2col.json --maxsat
        echo ">>> Extract with ILP-ACyc"
        timeout 20 ./target/debug/flexmatch ../tests/models/$relay_file rewrite.json data.json simpl.json im2col.json --new-ilp
        echo ">>> Extract with ILP-Topo"
        timeout 20 ./target/debug/flexmatch ../tests/models/$relay_file rewrite.json data.json simpl.json im2col.json --topo-sort-ilp
    done
done

mv extraction_stats.txt extraction_stats_im2col_with_simpl.txt