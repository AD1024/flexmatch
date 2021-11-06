mod rewrites;

use egg::{EGraph, Extractor, Language, LpCostFunction, LpExtractor, RecExpr, Runner};
use glenside::{
    extraction::AcceleratorCostFunction,
    language::{serialize_analysis_data, MyAnalysis, MyAnalysisData, RelayOperator},
};
use rewrites::{get_rewrite_from_string, im2col_rewrites, linear_rewrites};
use serde::Deserialize;
use serde_json;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::exit,
};
use tvm;

extern crate env_logger;
extern crate log;

use log::{debug, info};

#[derive(Deserialize, Clone, Debug)]
struct RewriteConfig {
    rewrites: HashMap<String, Box<[i32]>>,
    composites: HashMap<String, String>,
    compilers: HashMap<String, String>,
    debug_functions: HashMap<String, String>,
}

fn read_configs(flexmatch_home: &PathBuf, config_files: &[String]) -> Vec<RewriteConfig> {
    let mut result: Vec<RewriteConfig> = Vec::default();
    for config in config_files {
        let config_file = flexmatch_home.join(Path::new("configs").join(config));
        if let Ok(content) = std::fs::read_to_string(config_file) {
            result.push(serde_json::from_str(&content).unwrap());
            debug!("{:?}", result[result.len() - 1]);
        } else {
            panic!("failed to read {}", config)
        }
    }
    return result;
}

fn save_egraph_as_recexpr(
    egraph: &EGraph<glenside::language::Language, MyAnalysis>,
    rec_expr: &mut RecExpr<glenside::language::Language>,
) {
    let mut expr_map: BTreeMap<egg::Id, glenside::language::Language> = BTreeMap::new();
    for eclass in egraph.classes() {
        expr_map.insert(eclass.id, eclass.nodes[0].clone());
    }
    for (_id, expr) in expr_map.into_iter() {
        rec_expr.add(expr);
    }
}

fn save_expr_and_analysis(
    rec_expr_file: PathBuf,
    analysis_data_file: PathBuf,
    input_shapes: &HashMap<String, Vec<usize>>,
    best: &egg::RecExpr<glenside::language::Language>,
) {
    let json_dump = best.serialize();
    let output_file = PathBuf::from(env::current_dir().unwrap()).join(rec_expr_file);
    let _ = std::fs::write(output_file, json_dump.to_string()).unwrap();
    let mut egraph = EGraph::new(MyAnalysis {
        name_to_shape: input_shapes.clone(),
    });
    let (_, id_map) = egraph.add_expr_with_record(&best);
    egraph.rebuild();
    let mut native_map = HashMap::new();
    for (k, v) in id_map.into_iter() {
        native_map.insert(k, v);
    }
    let data_json_dump = serialize_analysis_data(&egraph, &native_map);
    let data_output = PathBuf::from(env::current_dir().unwrap()).join(analysis_data_file);
    let _ = std::fs::write(data_output, data_json_dump.to_string()).unwrap();
}

fn main() {
    env_logger::init();
    let args = env::args().collect::<Vec<_>>();
    let flexmatch_home = PathBuf::from(env::var("FLEXMATCH_HOME").unwrap());
    if args.len() < 3 {
        println!("flexmatch src_file recexpr_json data_json [config.json]+ Optional[--ilp]");
        exit(0);
    } else {
        let source_file = &args[1];
        let output_file = PathBuf::from(&args[2]);
        let analysis_data_file = PathBuf::from(&args[3]);
        let use_ilp = &args[args.len() - 1] == "--ilp";
        let config_files = if use_ilp {
            &args[4..args.len() - 1]
        } else {
            &args[4..]
        };

        let aggregated_configs = read_configs(&flexmatch_home, config_files);
        let mut rewrites = vec![];
        let mut rewrite_set = HashSet::new();
        debug!("{:?}", aggregated_configs);
        for config in aggregated_configs.iter() {
            for (rws, rw_args) in config.rewrites.iter() {
                if rewrite_set.contains(rws) {
                    continue;
                }
                rewrite_set.insert(rws.clone());
                match rws.as_str() {
                    "im2col" => rewrites.extend(im2col_rewrites()),
                    "linear-rewrites" => rewrites.extend(linear_rewrites()),
                    _ => rewrites.push(get_rewrite_from_string(rws, rw_args)),
                }
            }
        }
        let relay_src = fs::read_to_string(PathBuf::from(source_file)).unwrap();
        let module: tvm::ir::module::IRModule =
            tvm::ir::module::IRModule::parse("", relay_src).unwrap();
        info!("Compiling to Glenside");
        let (expr, shape_info, equiv_worklist) = glenside::language::from_relay::from_relay(
            &module,
            false,
            &vec![RelayOperator::RelaySigmoid],
        );
        let mut env = HashMap::default();
        for (name, shape) in &shape_info {
            env.insert(name.clone(), shape.clone());
        }
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });
        info!("Merging equivalent expressions");
        let (id, id_map) = egraph.add_expr_with_record(&expr);
        for (left, right) in equiv_worklist {
            if let (Some(&new_left), Some(&new_right)) = (id_map.get(&left), id_map.get(&right)) {
                egraph.union(new_left, new_right);
            }
        }
        egraph.rebuild();
        info!("Running Equality Saturation");
        let mut runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .with_time_limit(std::time::Duration::from_secs(20))
            .with_node_limit(900000)
            .with_iter_limit(100)
            .run(&rewrites);
        info!("EqSat Complete");
        // propagate accelerator calls
        let mut analysis_worklist = runner.egraph.analysis_update_worklist(|a, _p| {
            match a {
                glenside::language::MyAnalysisData::AccessPattern(access) => access.contains_accelerator_calls,
                _ => false,
            }
        });
        while analysis_worklist.len() > 0 {
            for (ids, _ref_id) in analysis_worklist.iter() {
                for id in ids.iter().cloned() {
                    let analysis_data = &mut runner.egraph[id].data;
                    match analysis_data {
                        glenside::language::MyAnalysisData::AccessPattern(access) => {
                            access.contains_accelerator_calls = true;
                        }
                        _ => ()
                    }
                }
            }
            analysis_worklist.clear();
            analysis_worklist.extend(runner.egraph.analysis_update_worklist(|a, parents| {
                match a {
                    glenside::language::MyAnalysisData::AccessPattern(access) => {
                        if access.contains_accelerator_calls {
                            parents.iter().map(|x| x.1).any(|pid| {
                                match &runner.egraph[pid].data {
                                    glenside::language::MyAnalysisData::AccessPattern(access) => !access.contains_accelerator_calls,
                                    _ => false,
                                }
                            })
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            }));
        }
        if !use_ilp {
            info!("Extraction without ILP");
            let extractor = Extractor::new(&runner.egraph, AcceleratorCostFunction {});
            let (_cost, best) = extractor.find_best(id);
            save_expr_and_analysis(output_file, analysis_data_file, &env, &best);
        } else {
            info!("Extracting with ILP solver");
            // egg ilp extraction
            struct LpAcceleratorCostFn;
            impl LpCostFunction<glenside::language::Language, MyAnalysis> for LpAcceleratorCostFn {
                fn node_cost(&mut self, egraph: &EGraph<glenside::language::Language, MyAnalysis>, _eclass: egg::Id, enode: &glenside::language::Language) -> f64 {
                    return get_node_weights(enode, egraph);
                }
            }
            let extractor = LpExtractor::new(&runner.egraph, LpAcceleratorCostFn {});
            let expr = extractor.solve(id);
            save_expr_and_analysis(output_file, analysis_data_file, &env, &expr);
            /*

            // The following extraction strategy is borrowed from Glenside ISCA demo
            #[cfg(feature = "cplex")]
            {
                use rplex::*;
                info!("Extraction with ILP solver");
                let mut cplex_env = Env::new().unwrap();
                cplex_env.set_param(EnvParam::ScreenOutput(true)).unwrap();
                // cplex_env.set_param(EnvParam::RelativeGap(0.9)).unwrap();
                // cplex_env.set_param(EnvParam::Threads(60)).unwrap();
                // cplex_env.set_param(EnvParam::MIPStrategyProbe(3)).unwrap();
                // Finds "hidden" solutions -- seems to work well with our problems, where
                // CPLEX struggles to even find one solution.
                // cplex_env.set_param(EnvParam::MIPEmphasis(4)).unwrap();
                cplex_env
                    .set_param(EnvParam::MIPLimitsSolutions(1))
                    .unwrap();
                // Deterministic time limit in "ticks"
                cplex_env
                    .set_param(EnvParam::DetTimeLimit(100000.0))
                    .unwrap();
                info!("Root eclass analysis: {:?}", runner.egraph[id].data);
                let mut model = glenside::extraction::ilp::create_generic_egraph_lp_model(
                    &cplex_env,
                    &runner.egraph,
                    |node, id, egraph| true,
                    &[id],
                    "ilp-extraction",
                );
                let mut costs = Constraint::new(
                    ConstraintType::Eq, /*ignored*/
                    0.0,                /*ignored*/
                    "costs",
                );
                for (_, var) in model.bq_vars.iter() {
                    costs.add_wvar(WeightedVariable::new_idx(*var, 1.0));
                }
                for (_, var) in model.topo_sort_vars.iter() {
                    costs.add_wvar(WeightedVariable::new_idx(*var, 0.0));
                }
                for (&node, var) in model.bn_vars.iter() {
                    let weight = get_node_weights(node, &runner.egraph);
                    costs.add_wvar(WeightedVariable::new_idx(*var, weight));
                }
                model
                    .problem
                    .set_objective(ObjectiveType::Minimize, costs)
                    .unwrap();
                info!("objective set");

                info!("ilp problem created, beginning solving...");
                let result = model.problem.solve().unwrap();
                info!("ilp problem solved");

                let (expr, _old_id_to_new_id_map) =
                    glenside::extraction::ilp::extract_single_expression(
                        &model,
                        &result.variables,
                        EGraph::new(MyAnalysis {
                            name_to_shape: env.clone(),
                        }),
                    );
                let mut rec_expr = RecExpr::default();
                save_egraph_as_recexpr(&expr, &mut rec_expr);
                info!("Save RecExpr to {:?}", output_file);
                save_expr_and_analysis(output_file, analysis_data_file, &env, &rec_expr);
            } */
        }
    }
}

fn check_accelerator_call_by_eid(
    ch_id: &egg::Id,
    egraph: &EGraph<glenside::language::Language, MyAnalysis>,
) -> bool {
    match &egraph[*ch_id].data {
        MyAnalysisData::AccessPattern(access) => access.contains_accelerator_calls,
        _ => false,
    }
}

fn get_node_weights(node: &glenside::language::Language, egraph: &EGraph<glenside::language::Language, MyAnalysis>) -> f64 {
        if node
            .children()
            .iter()
            .any(|ch_id| check_accelerator_call_by_eid(ch_id, egraph))
        {
            -10.0
        } else {
            match node {
                // We only consider accelerator calls and relay operators for now when
                // extracting a model
                glenside::language::Language::AcceleratorCall(_) => -(egraph.total_size() as f64),
                glenside::language::Language::Access(_)
                | glenside::language::Language::List(_)
                | glenside::language::Language::Shape(_)
                | glenside::language::Language::Usize(_)
                | glenside::language::Language::AccessShape(_)
                | glenside::language::Language::AcceleratorFunc(_)
                | glenside::language::Language::Symbol(_)
                | glenside::language::Language::RelayOperatorCall(_)
                | glenside::language::Language::PadType(_)
                | glenside::language::Language::Int32(_)
                | glenside::language::Language::AccessTensor(_) => 1.0,
                glenside::language::Language::RelayOperator(op) => match op {
                    glenside::language::RelayOperator::RelayReshape
                    | glenside::language::RelayOperator::RelayBatchFlatten => 1.0,
                    glenside::language::RelayOperator::RelayAdd
                    | glenside::language::RelayOperator::RelayMaximum
                    | glenside::language::RelayOperator::RelayMinimum
                    | glenside::language::RelayOperator::RelayMean
                    | glenside::language::RelayOperator::RelayMultiply
                    | glenside::language::RelayOperator::RelayErf
                    | glenside::language::RelayOperator::RelayReLU
                    | glenside::language::RelayOperator::RelaySoftmax
                    | glenside::language::RelayOperator::RelayBiasAdd
                    | glenside::language::RelayOperator::RelaySigmoid
                    | glenside::language::RelayOperator::RelayLeakyReLU => 2.0,
                    glenside::language::RelayOperator::RelayDense => 3.0,
                    glenside::language::RelayOperator::RelayConv1D
                    | glenside::language::RelayOperator::RelayConv2D
                    | glenside::language::RelayOperator::RelayUpSampling
                    | glenside::language::RelayOperator::RelayBatchNormInference
                    | glenside::language::RelayOperator::RelayAvgPool2D
                    | glenside::language::RelayOperator::RelayGlobalAvgPool2D
                    | glenside::language::RelayOperator::RelayMaxPool2D => 4.0,
                },
                glenside::language::Language::AccessTranspose(_)
                | glenside::language::Language::RelayKernelLayout(_)
                | glenside::language::Language::RelayActivationLayout(_)
                | glenside::language::Language::NotNanFloat64(_)
                | glenside::language::Language::AccessPad(_)
                | glenside::language::Language::AccessFlatten(_)
                | glenside::language::Language::AccessWindows(_)
                | glenside::language::Language::AccessBroadcast(_)
                | glenside::language::Language::AccessSqueeze(_) => 2.0,

                glenside::language::Language::AccessInsertAxis(_)
                | glenside::language::Language::AccessCartesianProduct(_) => 5.0,

                glenside::language::Language::Compute(_) => 1.0,
                glenside::language::Language::AccessReshape(_) => 10000.0,
                glenside::language::Language::ComputeType(compute_type) => {
                    match compute_type {
                        glenside::language::ComputeType::ReLU
                        | glenside::language::ComputeType::ReduceMean
                        | glenside::language::ComputeType::ElementwiseAdd
                        | glenside::language::ComputeType::ElementwiseMul
                        | glenside::language::ComputeType::DotProduct
                        | glenside::language::ComputeType::ReduceMax => 10000.0,
                        _ => 100.0,
                    }
                }
                glenside::language::Language::AccessPair(_) => 10000.0,
                _ => 500.0,
            }
        }
}

fn filter_nodes(
    node: &glenside::language::Language,
    id: egg::Id,
    egraph: &EGraph<glenside::language::Language, MyAnalysis>,
) -> bool {
    if let glenside::language::Language::AcceleratorCall(_) = &node {
        return true;
    }
    let contains_accel_call =
        if let glenside::language::MyAnalysisData::AccessPattern(access) = &egraph[id].data {
            access.contains_accelerator_calls
        } else {
            false
        };
    if contains_accel_call {
        let result = node
            .children()
            .iter()
            .any(|cid| check_accelerator_call_by_eid(cid, egraph));
        if !result {
            debug!("Say no to {:?} because it's not an accelerator call from {:?}", node, &egraph[id]);
            return false;
        } else {
            return true;
        }
    }
    if egraph[id].nodes.iter().any(|expr| match expr {
        &glenside::language::Language::RelayOperatorCall(_) => true,
        &_ => false,
    }) {
        match node {
            glenside::language::Language::RelayOperatorCall(_)
            | glenside::language::Language::AccessTensor(_)
            | glenside::language::Language::Access(_) => true,
            _ => {
                debug!(
                    "Say no to {:?} because {:?} has relay nodes",
                    node, &egraph[id].nodes
                );
                false
            }
        }
    } else {
        true
    }
}
