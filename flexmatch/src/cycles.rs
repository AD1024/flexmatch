use egg::{Analysis, EGraph, Id, Language};
use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
};

#[derive(Debug)]
pub struct HyperGraph {
    /// hyper-edges from an e-class to its neighbors
    /// multiple e-nodes might be connecting to the same e-node
    /// therefore, we collapse them together and record the corresponding
    /// e-nodes (as usize, representing the variables in the MAXSAT problem)
    edges: HashMap<usize, HashMap<usize, HashSet<usize>>>,
    nodes: HashSet<usize>,
    ids_to_nodes: HashMap<Id, usize>,
    nodes_to_ids: HashMap<usize, Id>,
    num_nodes: usize,
}

impl HyperGraph {
    pub fn new() -> Self {
        HyperGraph {
            edges: HashMap::new(),
            nodes: HashSet::new(),
            ids_to_nodes: HashMap::new(),
            nodes_to_ids: HashMap::new(),
            num_nodes: 0,
        }
    }

    pub fn contains(&self, eclass: &Id) -> bool {
        self.ids_to_nodes.contains_key(eclass)
    }

    pub fn edges(&self, eclass: &Id) -> Option<HashMap<Id, &HashSet<usize>>> {
        if self.contains(eclass) {
            let mut result = HashMap::new();
            for (to, enodes) in self.edges[&self.ids_to_nodes[eclass]].iter() {
                result.insert(self.nodes_to_ids[to], enodes);
            }
            Some(result)
        } else {
            None
        }
    }

    pub fn nodes(&self) -> HashSet<Id> {
        self.nodes.iter().map(|x| self.nodes_to_ids[x]).collect()
    }

    pub fn dump(&self, path: PathBuf) {
        // let f = std::fs::
        let mut graph_str = String::from("");
        for (u, v) in self.edges.iter() {
            for w in v.keys() {
                graph_str += &format!("{} {}\n", u, w);
            }
        }
        std::fs::write(path, graph_str);
    }

    fn add_node(&mut self, k: Id) {
        let node_id = self.num_nodes;
        self.ids_to_nodes.insert(k, node_id);
        self.nodes_to_ids.insert(node_id, k);
        self.edges.insert(node_id, HashMap::new());
        self.nodes.insert(node_id);
        self.num_nodes += 1;
    }

    fn connect(&mut self, from: &Id, to: &Id, enode: usize) {
        if !self.contains(from) {
            self.add_node(*from);
        }
        if !self.contains(to) {
            self.add_node(*to);
        }
        let from = &self.ids_to_nodes[from];
        let to = &self.ids_to_nodes[to];
        if !self.edges[from].contains_key(to) {
            self.edges
                .get_mut(from)
                .unwrap()
                .insert(*to, HashSet::from([enode]));
        } else {
            self.edges
                .get_mut(from)
                .unwrap()
                .get_mut(to)
                .unwrap()
                .insert(enode);
        }
    }

    pub fn stats(&self) {
        println!("Num Nodes: {}", self.nodes.len());
        println!(
            "Num Edges: {}",
            self.edges.values().map(|m| m.len()).sum::<usize>()
        );
    }

    pub fn neighbors(&self, u: &Id) -> Vec<&Id> {
        if self.contains(u) {
            self.edges[&self.ids_to_nodes[u]]
                .keys()
                .map(|x| &self.nodes_to_ids[x])
                .collect()
        } else {
            vec![]
        }
    }

    pub fn get_node_by_id(&self, id: &Id) -> usize {
        self.ids_to_nodes[id]
    }

    pub fn get_id_by_node(&self, node: usize) -> Id {
        self.nodes_to_ids[&node]
    }

    pub fn remove_node_raw(&mut self, node: usize) {
        if self.nodes.contains(&node) {
            self.edges.remove(&node);
            for (k, v) in self.edges.iter_mut() {
                v.remove(&node);
            }
            self.nodes.remove(&node);
        }
    }

    pub fn remove_node(&mut self, node: &Id) {
        let node_id = &self.ids_to_nodes[node];
        if self.contains(node) {
            self.edges.remove(node_id);
            for (k, v) in self.edges.iter_mut() {
                v.remove(node_id);
            }
            self.nodes.remove(node_id);
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn subgraph<'a, T: Iterator<Item = &'a Id>>(&self, nodes: T) -> Self {
        let mut graph = HyperGraph::new();
        let node_set: HashSet<&Id> = nodes.collect();
        for &n in node_set.iter() {
            assert!(self.contains(n));
            let edges = self.edges(n).unwrap();
            for (neighbor, enodes) in edges.iter() {
                if !node_set.contains(neighbor) {
                    continue;
                }
                for enode in enodes.iter() {
                    graph.connect(n, neighbor, *enode);
                }
            }
        }
        graph
    }
}

pub fn to_hypergraph<L, N>(
    root: &Id,
    egraph: &EGraph<L, N>,
    node_vars: &HashMap<L, usize>,
    hgraph: &mut HyperGraph,
) where
    L: Language,
    N: Analysis<L>,
{
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_front(*root);
    visited.insert(*root);
    while !queue.is_empty() {
        let front = queue.pop_front().unwrap();
        for node in egraph[front].nodes.iter() {
            for ch in node.children() {
                hgraph.connect(&front, ch, node_vars[node]);
                if !visited.contains(ch) {
                    visited.insert(*ch);
                    queue.push_back(*ch);
                }
            }
        }
    }
}

pub mod scc {
    use itertools::Itertools;

    use super::*;

    fn scc_impl(
        v: &Id,
        graph: &HyperGraph,
        num: &mut HashMap<Id, usize>,
        low: &mut HashMap<Id, usize>,
        stack: &mut Vec<Id>,
        visited: &mut HashSet<Id>,
        onstack: &mut HashSet<Id>,
        idx: &mut usize,
        scc: &mut Vec<Vec<Id>>,
    ) {
        num.insert(*v, *idx);
        low.insert(*v, *idx);
        *idx += 1;
        visited.insert(*v);
        stack.push(*v);
        onstack.insert(*v);

        for u in graph.neighbors(v) {
            if !visited.contains(u) {
                // a tree edge
                scc_impl(u, graph, num, low, stack, visited, onstack, idx, scc);
                if low[v] > low[u] {
                    low.insert(*v, low[u]);
                }
            } else if onstack.contains(u) {
                // back edge
                if low[v] > num[u] {
                    low.insert(*v, num[u]);
                }
            }
        }
        if low[v] == num[v] {
            // found an scc
            let mut scc_found = Vec::new();
            let mut scc_rt = stack.pop().unwrap();
            onstack.remove(&scc_rt);
            while scc_rt != *v {
                scc_found.push(scc_rt);
                scc_rt = stack.pop().unwrap();
                onstack.remove(&scc_rt);
            }
            scc_found.push(scc_rt);
            scc.push(scc_found);
        }
    }

    pub fn scc(graph: &HyperGraph) -> Vec<Vec<Id>> {
        let mut num = HashMap::new();
        let mut low = HashMap::new();
        let mut visited = HashSet::new();
        let mut processed = HashSet::new();
        let mut stack = Vec::new();
        let mut idx = 0;
        let mut scc = Vec::new();
        for v in graph.nodes().iter().sorted() {
            if !visited.contains(v) {
                scc_impl(
                    v,
                    graph,
                    &mut num,
                    &mut low,
                    &mut stack,
                    &mut visited,
                    &mut processed,
                    &mut idx,
                    &mut scc,
                )
            }
        }
        return scc;
    }
}

pub mod johnson {
    use itertools::Itertools;

    use super::*;

    fn johnson_alg_impl(v: Id, hgraph: &HyperGraph, cycles: &mut Vec<Vec<Id>>) {
        let mut path = vec![v];
        let mut stack = Vec::new();
        stack.push(hgraph.neighbors(&v));
        let start = v;
        let mut closed = vec![false];
        let mut blocked = HashSet::new();
        blocked.insert(v);
        let mut blocked_dict: HashMap<Id, Vec<Id>> = HashMap::new();
        while !stack.is_empty() {
            let next = stack.pop().unwrap();
            let mut f = false;
            for w in next {
                f = true;
                if *w == start {
                    cycles.push(path.clone());
                    *closed.last_mut().unwrap() = true;
                } else if !blocked.contains(w) {
                    path.push(*w);
                    blocked.insert(*w);
                    closed.push(false);
                    stack.push(hgraph.neighbors(w));
                    break;
                }
            }
            if !f {
                stack.pop();
                let w = path.pop().unwrap();
                if closed.pop().unwrap() {
                    if !closed.is_empty() {
                        *closed.last_mut().unwrap() = true;
                    }
                    let mut unblock_chain = vec![w];
                    while !unblock_chain.is_empty() {
                        let v = unblock_chain.pop().unwrap();
                        if blocked.contains(&v) {
                            blocked.remove(&v);
                            unblock_chain = [
                                unblock_chain,
                                blocked_dict.get(&v).unwrap_or(&vec![]).clone(),
                            ]
                            .concat();
                            blocked_dict.get_mut(&v).unwrap_or(&mut vec![]).clear();
                        }
                    }
                } else {
                    for u in hgraph.neighbors(&w) {
                        if !blocked_dict.contains_key(u) {
                            blocked_dict.insert(*u, vec![]);
                        }
                        blocked_dict.get_mut(u).unwrap().push(w);
                    }
                }
            }
        }
    }

    pub fn find_cycles(hgraph: &HyperGraph) -> Vec<Vec<Id>> {
        let mut scc = scc::scc(hgraph)
            .into_iter()
            .filter(|c| c.len() >= 2)
            .collect::<Vec<_>>();
        println!("SCC len: {}", scc.len());
        let mut cycles = Vec::new();
        while !scc.is_empty() {
            let cur_scc = scc.pop().unwrap();
            let mut subgraph = hgraph.subgraph(cur_scc.iter());
            for i in 0..cur_scc.len() {
                let v = subgraph.get_id_by_node(i);
                johnson_alg_impl(v, &subgraph, &mut cycles);
                subgraph.remove_node_raw(i);
            }
        }
        cycles
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_scc_johnson() {
        fn node(x: usize) -> Id {
            Id::from(x)
        }
        fn connect(graph: &mut HyperGraph, u: usize, v: usize) {
            graph.connect(&node(u), &node(v), 0);
        }
        let graph = &mut HyperGraph::new();
        connect(graph, 1, 2);
        connect(graph, 2, 3);
        connect(graph, 3, 1);

        connect(graph, 2, 4);
        connect(graph, 4, 5);
        connect(graph, 5, 6);
        connect(graph, 6, 5);

        connect(graph, 10, 7);
        connect(graph, 10, 8);
        connect(graph, 7, 8);
        connect(graph, 8, 9);
        connect(graph, 9, 10);

        connect(graph, 7, 5);
        connect(graph, 8, 6);
        connect(graph, 4, 4);
        let sccs = scc::scc(graph);
        assert_eq!(sccs.len(), 4);
        let ans: Vec<Vec<Id>> = vec![vec![6, 5], vec![7, 8, 9, 10], vec![4], vec![1, 2, 3]]
            .into_iter()
            .map(|v| v.into_iter().map(node).collect())
            .collect::<Vec<_>>();
        fn all_match<T>(l: &Vec<T>, r: &Vec<T>) -> bool
        where
            T: Eq + PartialEq,
        {
            for (x, y) in l.iter().zip(r.iter()) {
                if !r.contains(x) {
                    return false;
                }
                if !l.contains(y) {
                    return false;
                }
            }
            return true;
        }
        for r in ans.iter() {
            let mut matched = false;
            for l in sccs.iter() {
                if all_match(l, r) {
                    matched = true;
                    break;
                }
            }
            assert!(matched);
        }
        let cycles = johnson::find_cycles(graph);
        println!("{:?}", cycles);
    }
}
