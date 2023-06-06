use egg::{Analysis, EGraph, Id, Language};
use itertools::Itertools;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug)]
pub struct HyperGraph {
    /// hyper-edges from an e-class to its neighbors
    /// multiple e-nodes might be connecting to the same e-node
    /// therefore, we collapse them together and record the corresponding
    /// e-nodes (as usize, representing the variables in the MAXSAT problem)
    edges: HashMap<Id, HashMap<Id, HashSet<usize>>>,
    nodes: HashSet<Id>,
}

impl HyperGraph {
    pub fn new() -> Self {
        HyperGraph {
            edges: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    pub fn contains(&self, eclass: &Id) -> bool {
        self.edges.contains_key(eclass)
    }

    pub fn edges(&self, eclass: &Id) -> Option<&HashMap<Id, HashSet<usize>>> {
        if self.contains(eclass) {
            Some(&self.edges[eclass])
        } else {
            None
        }
    }

    fn add_node(&mut self, k: Id) {
        self.edges.insert(k, HashMap::new());
        self.nodes.insert(k);
    }

    fn connect(&mut self, from: &Id, to: &Id, enode: usize) {
        if !self.contains(from) {
            self.add_node(*from);
        }
        if !self.contains(to) {
            self.add_node(*to);
        }
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
            self.edges[u].keys().collect()
        } else {
            vec![]
        }
    }

    pub fn subgraph<'a, T: Iterator<Item = &'a Id>>(&self, nodes: T) -> Self {
        let mut graph = HyperGraph::new();
        let node_set: HashSet<&Id> = nodes.collect();
        for n in node_set.iter() {
            if self.contains(n) {
                graph.edges.insert(
                    **n,
                    self.edges
                        .get(n)
                        .unwrap()
                        .iter()
                        .filter(|(n, _)| node_set.contains(n))
                        .map(|(&x, y)| (x, y.clone()))
                        .collect::<HashMap<_, _>>(),
                );
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
        for v in graph.nodes.iter().sorted() {
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

    fn unblock(v: Id, blocked: &mut HashSet<Id>, block_list: &mut HashMap<Id, HashSet<Id>>) {
        blocked.remove(&v);
        while !block_list[&v].is_empty() {
            let blocked_set = block_list.get_mut(&v).unwrap();
            let worklist: Vec<_> = blocked_set.drain().collect();
            for i in worklist {
                if blocked.contains(&i) {
                    unblock(i, blocked, block_list);
                }
            }
        }
    }

    fn johnson_naive(
        start: Id,
        v: Id,
        graph: &HyperGraph,
        cycles: &mut Vec<Vec<Id>>,
        blocked: &mut HashSet<Id>,
        stack: &mut Vec<Id>,
        block_list: &mut HashMap<Id, HashSet<Id>>,
    ) -> bool {
        stack.push(v);
        blocked.insert(v);
        let mut f = false;
        for w in graph.neighbors(&v) {
            if *w == start {
                cycles.push(stack.clone());
                f = true;
            } else if *w > start && !blocked.contains(w) {
                f = johnson_naive(start, *w, graph, cycles, blocked, stack, block_list);
            }
        }
        if f {
            unblock(v, blocked, block_list);
        } else {
            for w in graph.neighbors(&v) {
                if !block_list[w].contains(&v) {
                    block_list.get_mut(w).unwrap().insert(v);
                }
            }
        }
        stack.pop();
        f
    }

    pub fn find_cycles(hgraph: &HyperGraph) -> Vec<Vec<Id>> {
        // let mut scc = scc::scc(hgraph);
        let mut cycles = Vec::new();
        // println!("SCC len: {}", scc.len());
        // while !scc.is_empty() {
        //     let cur_scc = scc.pop().unwrap();
        //     let subgraph = hgraph.subgraph(cur_scc.iter());
        //     let v = cur_scc[0];
        //     johnson_alg_impl(v, &subgraph, &mut cycles);
        // }
        let mut stack = Vec::new();
        let mut blocked = HashSet::new();
        let mut block_list: HashMap<Id, HashSet<Id>> = HashMap::new();
        let nodes = hgraph.nodes.iter().sorted().collect::<Vec<_>>();
        let mut i = 0;
        while i < nodes.len() {
            let start = nodes[i];
            for j in i..nodes.len() {
                blocked.remove(nodes[j]);
                block_list.insert(*nodes[j], HashSet::new());
            }
            println!("Node {}", i);
            johnson_naive(
                *start,
                *start,
                hgraph,
                &mut cycles,
                &mut blocked,
                &mut stack,
                &mut block_list,
            );
            i += 1;
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
