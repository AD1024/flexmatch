# flexmatch
Flexible Matching via Equality Saturation

# Workflow
1. Having patterns defined in Relay
2. Run EqSat on each pattern and get the corresponding EGraph
3. Pattern match on a given model using `EGraphMatcher`
4. Annotate matched region with BYOC-style function annotations for the codegen