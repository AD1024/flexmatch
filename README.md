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