mod rewrites;

use egg::{EGraph, Extractor, Runner};
use glenside::{
    extraction::AcceleratorCostFunction,
    language::{serialize_analysis_data, MyAnalysis},
};
use rewrites::{get_rewrite_from_string, im2col_rewrites, linear_rewrites};
use std::{collections::HashMap, env, fs, path::PathBuf, process::exit};
use tvm;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        println!("flexmatch src_file recexpr_json data_json rewrites+");
        exit(0);
    } else {
        let source_file = &args[1];
        let output_file = &args[2];
        let analysis_data_file = &args[3];
        let mut rewrites = vec![];
        for rws in args.iter().skip(4).cloned() {
            match rws.as_str() {
                "im2col" => rewrites.extend(im2col_rewrites()),
                "linear-rewrites" => rewrites.extend(linear_rewrites()),
                _ => rewrites.push(get_rewrite_from_string(rws)),
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
        let mut native_map = HashMap::new();
        for (k, v) in id_map.into_iter() {
            native_map.insert(k, v);
        }
        let data_json_dump = serialize_analysis_data(&egraph, &native_map);
        let data_output = PathBuf::from(env::current_dir().unwrap()).join(analysis_data_file);
        let _ = std::fs::write(data_output, data_json_dump.to_string()).unwrap();
    }
}
