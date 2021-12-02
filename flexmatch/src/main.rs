mod rewrites;

use egg::{EGraph, Extractor, Language, Runner};
use glenside::{
    extraction::AcceleratorCostFunction,
    language::{serialize_analysis_data, MyAnalysis, MyAnalysisData, RelayOperator},
};
use rewrites::{get_rewrite_from_string, im2col_rewrites, linear_rewrites};
use serde::Deserialize;
use serde_json;
use simge::{
    from_glenside,
    memory::{DRAM, SRAM},
    sim::*,
};
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
    out_dtypes: HashMap<String, String>,
}

#[derive(Deserialize, Clone, Debug)]
struct SramConfigs {
    srams: HashMap<String, usize>,
    policy: String,
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

fn main() {
    env_logger::init();
    let args = env::args().collect::<Vec<_>>();
    let flexmatch_home = PathBuf::from(env::var("FLEXMATCH_HOME").unwrap());
    if args.len() < 3 {
        println!("flexmatch src_file recexpr_json data_json [config.json]+ Optional[--ilp]");
        exit(0);
    } else {
        let source_file = &args[1];
        let sram_configs = PathBuf::from(&args[2]);
        let use_ilp = &args[args.len() - 1] == "--ilp";
        let config_files = if use_ilp {
            &args[3..args.len() - 1]
        } else {
            &args[3..]
        };

        let sram_config_path = flexmatch_home.join(Path::new("configs").join(sram_configs.clone()));
        let mut sram_config = None;
        if let Ok(content) = std::fs::read_to_string(sram_config_path) {
            sram_config = Some(serde_json::from_str::<SramConfigs>(&content).unwrap());
        } else {
            panic!("failed to read {:?}", sram_configs);
        }

        let aggregated_configs = read_configs(&flexmatch_home, config_files);
        let mut rewrites = vec![];
        let mut rewrite_set = HashSet::new();
        debug!("{:?}", aggregated_configs);
        for config in aggregated_configs.iter() {
            for (rws, rw_args) in config.rewrites.iter() {
                if rewrite_set.contains(rws) {
                    continue;
                }
                info!("Adding rewrite: {:?} with args {:?}", rws, rw_args);
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
        let (expr, shape_info, dtype_info, equiv_worklist) =
            glenside::language::from_relay::from_relay(
                &module,
                false,
                &vec![
                    RelayOperator::RelaySigmoid,
                    RelayOperator::RelayAvgPool2D,
                    RelayOperator::RelayTanh,
                    RelayOperator::RelayLogSoftmax,
                    RelayOperator::RelayAdd,
                    RelayOperator::RelayStridedSlice,
                    RelayOperator::RelayReLU,
                ],
            );
        let mut env = HashMap::default();
        for (name, shape) in &shape_info {
            env.insert(name.clone(), shape.clone());
        }
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
            name_to_dtype: dtype_info.iter().cloned().collect(),
        });
        info!("Merging equivalent expressions");
        let (root_expr, id_map) = egraph.add_expr_with_record(&expr);
        for (left, right) in equiv_worklist {
            if let (Some(&new_left), Some(&new_right)) = (id_map.get(&left), id_map.get(&right)) {
                egraph.union(new_left, new_right);
            }
        }
        egraph.rebuild();
        info!("Running Equality Saturation");
        let mut runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .with_time_limit(std::time::Duration::from_secs(5))
            .with_node_limit(100000)
            .with_iter_limit(45)
            .run(&rewrites);
        info!("EqSat Complete");
        let root_expr = runner.egraph.find(root_expr);
        // propagate accelerator calls
        let mut analysis_worklist = runner.egraph.analysis_update_worklist(|a, _p| match a {
            glenside::language::MyAnalysisData::AccessPattern(access) => {
                access.contains_accelerator_calls
            }
            _ => false,
        });
        while analysis_worklist.len() > 0 {
            for (ids, _ref_id) in analysis_worklist.iter() {
                for id in ids.iter().cloned() {
                    let analysis_data = &mut runner.egraph[id].data;
                    match analysis_data {
                        glenside::language::MyAnalysisData::AccessPattern(access) => {
                            access.contains_accelerator_calls = true;
                        }
                        _ => (),
                    }
                }
            }
            analysis_worklist.clear();
            analysis_worklist.extend(runner.egraph.analysis_update_worklist(
                |a, parents| match a {
                    glenside::language::MyAnalysisData::AccessPattern(access) => {
                        if access.contains_accelerator_calls {
                            parents
                                .iter()
                                .map(|x| x.1)
                                .any(|pid| match &runner.egraph[pid].data {
                                    glenside::language::MyAnalysisData::AccessPattern(access) => {
                                        !access.contains_accelerator_calls
                                    }
                                    _ => false,
                                })
                        } else {
                            false
                        }
                    }
                    _ => false,
                },
            ));
        }
        info!("Root eclass analysis: {:?}", runner.egraph[root_expr].data);
        info!("Root eclass nodes: {:?}", runner.egraph[root_expr].nodes);
        if !use_ilp {
            info!("Extraction without ILP");
            let extractor = Extractor::new(
                &runner.egraph,
                AcceleratorCostFunction(runner.egraph.total_size() as f64),
            );
            let (_cost, best) = extractor.find_best(root_expr);
            // println!("{:?}\n{}", best.nodes, best.pretty(10));
            let mut new_egraph =
                egg::EGraph::<glenside::language::Language, MyAnalysis>::new(MyAnalysis {
                    name_to_shape: env.clone(),
                    name_to_dtype: dtype_info.iter().cloned().collect(),
                });
            let (_, id_map) = new_egraph.add_expr_with_record(&best);
            let (mut operators, _output_id) = from_glenside::compile_instruction(
                &egg::Id::from(best.nodes.len() - 1),
                &best,
                &mut HashMap::default(),
                &new_egraph,
                &id_map.into_iter().collect(),
            )
            .unwrap();
            let sram_config = sram_config.unwrap();
            let mut srams = sram_config
                .srams
                .into_iter()
                .map(|sram| (sram.0, SRAM::new(sram.1)))
                .collect::<HashMap<_, _>>();
            // srams.insert("vta".into(), vta_sram);
            // srams.insert("accel".into(), SRAM::new(1024));
            // println!("Operators: {:?}", operators);
            if "lru" == sram_config.policy.as_str() {
                let mut simulator = JitSim::new(simge::heuristics::LRU::new());
                simulator.run(
                    &mut operators,
                    &mut srams,
                    &mut DRAM::new(),
                    &mut HashSet::default(),
                );
            } else if "random" == sram_config.policy.as_str() {
                let mut simulator = JitSim::new(simge::heuristics::RandomEviction::new());
                simulator.run(
                    &mut operators,
                    &mut srams,
                    &mut DRAM::new(),
                    &mut HashSet::default(),
                );
            }
            info!(
                "Round Trip on SRAM: {} = {}",
                srams
                    .iter()
                    .map(|x| format!("{}({})", x.1.trip_count, x.0))
                    .collect::<Vec<_>>()
                    .join(" + "),
                srams.iter().map(|x| x.1.trip_count).sum::<usize>()
            );
        } else {
            // The following extraction strategy is borrowed from Glenside ISCA demo
            /*
            #[cfg(feature = "cplex")]
            {
                use rplex::*;
                info!("Extraction with ILP solver");
                let mut cplex_env = Env::new().unwrap();
                cplex_env.set_param(EnvParam::ScreenOutput(true)).unwrap();
                cplex_env
                    .set_param(EnvParam::MIPLimitsSolutions(1))
                    .unwrap();
                // Deterministic time limit in "ticks"
                cplex_env
                    .set_param(EnvParam::DetTimeLimit(6000000.0))
                    .unwrap();
                let mut model = glenside::extraction::ilp::create_generic_egraph_lp_model(
                    &cplex_env,
                    &runner.egraph,
                    |node, id, egraph| true && filter_nodes(node, id, egraph),
                    &[root_expr],
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
                    let weight = get_node_weights(node, runner.egraph.total_size() as f64);
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
                            name_to_dtype: dtype_info.iter().cloned().collect(),
                        }),
                    );
                let mut rec_expr = egg::RecExpr::default();
                save_egraph_as_recexpr(&expr, &mut rec_expr);
                save_expr_and_analysis(
                    output_file,
                    analysis_data_file,
                    &env,
                    &dtype_info.iter().cloned().collect(),
                    &rec_expr,
                );
            } */
        }
    }
}
