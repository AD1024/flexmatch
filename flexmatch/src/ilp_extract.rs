use std::{collections::HashMap, time::Instant};

use egg::{Analysis, EGraph, Id, Language, RecExpr};
use rand::Rng;
use rplex::{self, var, Problem, VariableValue, WeightedVariable};

fn get_all_cycles<'a, L, N>(
    egraph: &EGraph<L, N>,
    root: &Id,
    color: &mut HashMap<Id, usize>,
    path: &mut Vec<(Id, L)>,
    problem: &mut Problem<'a>,
    node_vars: &HashMap<L, usize>,
) where
    L: Language,
    N: Analysis<L>,
{
    if color.contains_key(root) && color[root] == 2 {
        return;
    }
    if color.contains_key(root) && color[root] == 1 {
        if let Some((idx, _)) = path.iter().enumerate().find(|(_, (id, _))| id == root) {
            let mut new_cycle = Vec::new();
            let subpath = path[idx..].to_vec();
            for (_, n) in subpath {
                new_cycle.push(node_vars[&n]);
            }
            let mut rng = rand::thread_rng();
            if new_cycle.len() == 1 {
                let mut constraint = rplex::Constraint::new(
                    rplex::ConstraintType::Eq,
                    0.0,
                    format!("cycle_{}_{}", root, rng.gen::<u64>()),
                );
                constraint.add_wvar(WeightedVariable::new_idx(new_cycle[0], 1.0));
                problem.add_constraint(constraint).unwrap();
            } else {
                for node_idx in egraph[*root].nodes.iter().map(|x| node_vars[x]) {
                    new_cycle[0] = node_idx;
                    // sum up <= len(new_cycle) - 1
                    let mut constraint = rplex::Constraint::new(
                        rplex::ConstraintType::LessThanEq,
                        new_cycle.len() as f64 - 1.0,
                        format!("cycle_{}_{}", root, rng.gen::<u64>()),
                    );
                    for node_idx in new_cycle.iter() {
                        constraint.add_wvar(WeightedVariable::new_idx(*node_idx, 1.0));
                    }
                    problem.add_constraint(constraint).unwrap();
                }
            }
            return;
        }
        panic!("Should have a cycle here: {}; path: {:?}", root, path);
    }
    color.insert(*root, 1);
    for node in egraph[*root].nodes.iter() {
        for ch in node.children() {
            path.push((*root, node.clone()));
            get_all_cycles(egraph, ch, color, path, problem, node_vars);
            path.pop();
        }
    }
    color.insert(*root, 2);
}

pub struct ILPProblem<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    pub node_vars: HashMap<L, usize>,
    pub problem: Problem<'a>,
    pub root: Id,
    pub egraph: &'a EGraph<L, N>,
}

impl<'a, L, N> ILPProblem<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    pub fn new(
        problem: Problem<'a>,
        egraph: &'a EGraph<L, N>,
        root: Id,
        node_vars: HashMap<L, usize>,
    ) -> Self {
        ILPProblem {
            problem,
            root,
            node_vars,
            egraph,
        }
    }

    pub fn solve(&mut self) -> (u128, f64, RecExpr<L>) {
        self.problem
            .set_objective_type(rplex::ObjectiveType::Minimize)
            .unwrap();
        let start = Instant::now();
        let solution = self.problem.solve().unwrap();
        let solve_time = start.elapsed().as_millis();
        let cost = solution.objective;

        let mut expr = RecExpr::default();
        let mut new_id_map = HashMap::new();
        let mut worklist = vec![self.root];
        let mut path: Vec<(Id, L)> = Vec::new();

        while let Some(&id) = worklist.last() {
            if new_id_map.contains_key(&id) {
                worklist.pop();
                path = path
                    .iter()
                    .cloned()
                    .filter(|(i, _)| *i != id)
                    .collect::<Vec<_>>();
                continue;
            }
            let mut not_found = true;
            for node in &self.egraph[id].nodes {
                let node_idx = self.node_vars[node];
                match solution.variables[node_idx] {
                    VariableValue::Binary(true) => {
                        path.push((id, node.clone()));
                        not_found = false;
                        if node.all(|c| new_id_map.contains_key(&c)) {
                            let new_id = expr.add(node.clone().map_children(|c| new_id_map[&c]));
                            new_id_map.insert(id, new_id);
                            path = path
                                .iter()
                                .cloned()
                                .filter(|(i, _)| *i != id)
                                .collect::<Vec<_>>();
                            worklist.pop();
                        } else {
                            let pending_work = node
                                .children()
                                .iter()
                                .filter(|&x| !new_id_map.contains_key(x))
                                .collect::<Vec<_>>();
                            for each in pending_work {
                                if let Some((idx, _)) =
                                    path.iter().enumerate().find(|(_, (id, _))| id == each)
                                {
                                    let subpath = path[idx..]
                                        .iter()
                                        .map(|(_, l)| l.clone())
                                        .collect::<Vec<_>>();
                                    panic!("Cycle detected at {}: {:?}", id, subpath);
                                }
                            }
                            worklist.extend(node.children());
                        }
                        break;
                    }
                    _ => continue,
                }
            }
            if not_found {
                panic!("No enode chosen for eclass {}", id);
            }
        }

        return (solve_time, cost, expr);
    }
}

fn node_name(idx: usize) -> String {
    format!("node_idx_{}", idx)
}

pub fn create_problem<'a, L, N, CF>(
    env: &'a rplex::Env,
    root: Id,
    egraph: &'a EGraph<L, N>,
    no_cycle: bool,
    mut cost_fn: CF,
) -> ILPProblem<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
    CF: FnMut(&EGraph<L, N>, egg::Id, &L) -> f64,
{
    let mut problem = Problem::new(&env, "egraph_ext").unwrap();
    let mut node_vars = HashMap::new();

    // create node_vars
    for eclass in egraph.classes() {
        for (idx, node) in eclass.nodes.iter().enumerate() {
            let name = node_name(idx);
            let cost = cost_fn(egraph, eclass.id, node);
            let var = var!(0.0 <= name <= 1.0 -> cost as Binary);
            node_vars.insert(node.clone(), problem.add_variable(var).unwrap());
        }
    }

    // root constraint
    let mut constraint = rplex::Constraint::new(rplex::ConstraintType::Eq, 1.0, "root constraint");
    for node_idx in egraph[root].nodes.iter().map(|x| node_vars[x]) {
        constraint.add_wvar(WeightedVariable::new_idx(node_idx, 1.0));
    }
    problem.add_constraint(constraint).unwrap();

    // children constraint
    for eclass in egraph.classes() {
        for node in egraph[eclass.id].nodes.iter() {
            let node_idx = node_vars[node];
            for (ch_idx, ch) in node.children().iter().enumerate() {
                let mut constraint = rplex::Constraint::new(
                    rplex::ConstraintType::GreaterThanEq,
                    0.0,
                    format!("{}_child_{}", node_idx, ch_idx),
                );
                constraint.add_wvar(WeightedVariable::new_idx(node_idx, -1.0));
                for ch_node_idx in egraph[*ch].nodes.iter().map(|x| node_vars[x]) {
                    constraint.add_wvar(WeightedVariable::new_idx(ch_node_idx, 1.0));
                }
                problem.add_constraint(constraint).unwrap();
            }
        }
    }

    if no_cycle {
        let mut color = HashMap::new();
        let mut path = Vec::new();
        get_all_cycles(
            egraph,
            &root,
            &mut color,
            &mut path,
            &mut problem,
            &node_vars,
        );
    }

    return ILPProblem::new(problem, egraph, root, node_vars);
}
