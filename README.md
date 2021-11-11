# flexmatch
Flexible Matching via Equality Saturation

# Workflow (Deprecated)
1. Having patterns defined in Relay
2. Run EqSat on each pattern and get the corresponding EGraph
3. Pattern match on a given model using `EGraphMatcher`
4. Annotate matched region with BYOC-style function annotations for the codegen

# Utilities
- `flexmatch`: Rust interfaces that call [glenside](https://github.com/gussmith23/glenside) and apis in (my fork of) [egg](https://github.com/AD1024/egg)
- `tests/run_eqsat`
    - Takes a **relay source file**, an output file name and config(s) (under `configs/`)
    - Run equality saturation on the given model with respect to rewrite rules in configs
    - Example: `python3 run_eqsat.py models/resmlp.relay resmlp im2col-rewrites linear-rewrites`
- `tests/compile_model`
    - Takes the **relay source file**, the output model and eclass analysis data json from EqSat
    - Compiles the rewritten model back to a relay executable model and saves to a file
    - Optional argument: `--debug`; instead of inserting accelerator calls, if this argument is passed, the equivalent relay function will be generated
    - Example: `python3 compile_model.py models/resmlp.relay resmlp-rewritten.relay resmlp-rewritten.json resmlp-data.json linear-rewrites im2col-rewrites --debug`

# End-to-end compilation validation
```bash
python3 validate_compilation.py model ([--configs CONFIGS+] | [--defaults]) [--use-ilp] [--debug]
```
- `model`: resnet18; efficientnet; max_pool2d; mobilenet; resmlp
- `configs`: hlscnn-conv2d; flexasr-maxpool; im2col; vta-dense; linear-rewrites
    - hlscnn-conv2d: `Conv2D` to HLSCNN
    - flexasr-maxpool: `Max_pool2D` to FlexASR
    - im2col: Convolutions to matmuls
    - linear-rewrites: `nn.Linear` to FlexASR
- `--defaults`: if turned oon, use default configs
- `--use-ilp`: Use CPLEX ILP Solver to extract the rewritten model
- `--debug`: Use debug functions to replace accelerator calls for debug purposes

# Rewrite Config Structure
- rewrites :: Dict[String, Array[Integer]]. rewrite rules to apply
- composites :: Dict[String (accelerator func names), String]. Compiler composite region annotations
- compiler :: Dict[String (accelerator func names), String]. which compiler to use
- debug_functions :: Dict[String (accelerator func names), String]. debug functions for corresponding accelerator calls
- out_dtypes :: Dict[String, String (dtype names)]. Output data type of the accelerator function.