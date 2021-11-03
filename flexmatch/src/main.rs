mod rewrites;

use egg::{EGraph, Extractor, Runner};
use glenside::{
    extraction::AcceleratorCostFunction,
    language::{serialize_analysis_data, MyAnalysis},
};
use rewrites::{get_rewrite_from_string, im2col_rewrites, linear_rewrites};
use serde::Deserialize;
use serde_json;
use std::{collections::{HashMap, HashSet}, env, fs, path::{Path, PathBuf}, process::exit};
use tvm;

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
            println!("{:?}", result[result.len() - 1]);
        } else {
            panic!("failed to read {}", config)
        }
    }
    return result;
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let flexmatch_home = PathBuf::from(env::var("FLEXMATCH_HOME").unwrap());
    if args.len() < 3 {
        println!("flexmatch src_file recexpr_json data_json [config.json]+");
        exit(0);
    } else {
        let source_file = &args[1];
        let output_file = &args[2];
        let analysis_data_file = &args[3];
        let config_files = &args[4..];

        let aggregated_configs = read_configs(&flexmatch_home, config_files);
        let mut rewrites = vec![];
        let mut rewrite_set = HashSet::new();
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
        let (expr, shape_info, equiv_worklist) =
            glenside::language::from_relay::from_relay(&module, false, &vec![]);
        let mut env = HashMap::default();
        for (name, shape) in &shape_info {
            env.insert(name.clone(), shape.clone());
        }
        let mut egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
        });
        let (id, id_map) = egraph.add_expr_with_record(&expr);
        for (left, right) in equiv_worklist {
            if let (Some(&new_left), Some(&new_right)) = (id_map.get(&left), id_map.get(&right)) {
                egraph.union(new_left, new_right);
            }
        }
        egraph.rebuild();
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(egraph)
            .with_time_limit(std::time::Duration::from_secs(5))
            .with_node_limit(500000)
            .with_iter_limit(40)
            .run(&rewrites);
        let extractor = Extractor::new(&runner.egraph, AcceleratorCostFunction {});
        let (_cost, best) = extractor.find_best(id);
        let json_dump = best.serialize();
        let output_file =
            PathBuf::from(env::current_dir().unwrap()).join(PathBuf::from(output_file));
        let _ = std::fs::write(output_file, json_dump.to_string()).unwrap();
        egraph = EGraph::new(MyAnalysis {
            name_to_shape: env.clone(),
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
}
