use std::collections::HashMap;
use std::io::BufRead;
use rsmaxsat::solver::MaxSATSolver;
use std::collections::VecDeque;

#[derive(Default,Debug)]
pub struct Graph {
    pub ngbh: HashMap<usize, Vec<usize>>,
}

impl Graph {
    fn add_node(&mut self, u: usize) {
        self.ngbh.entry(u).or_insert_with(Vec::new);
    }

    fn add_edge(&mut self, u: usize, v: usize) {
        self.ngbh.entry(u).or_insert_with(Vec::new).push(v);
        self.ngbh.entry(v).or_insert_with(Vec::new).push(u);
    }

    fn nnodes(&self) -> usize {
        return self.ngbh.len();
    }

    fn find_min_degree_vertex(&self) -> Option<(usize,usize)> {
        self.ngbh
            .iter()
            .min_by_key(|(_, vec)| vec.len())
            .map(|(&key, vec)| (key, vec.len()))
    }

    fn delete_node(&mut self, node_to_delete: usize) {
        self.ngbh.remove_entry(&node_to_delete);
        for vec in self.ngbh.values_mut() {
            vec.retain(|&x| x != node_to_delete);
        }
    }

    fn neighborhood(&self, u: usize) -> Vec<usize> {
        self.ngbh.get(&u).unwrap().to_vec()
    }
}

pub fn parse_graph() -> Graph {

    // graph initialization
    let mut graph: Graph = Default::default();
    let mut n0: usize;

    let stdin = std::io::stdin();

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.starts_with("p") {
            n0 = line
                .split_whitespace()
                .nth(2)
                .expect("number of vertics")
                .parse()
                .unwrap();

            for u in 1..=n0 {
                graph.add_node(u);
            }
            continue;
        }

        let a: usize = line
            .split_whitespace()
            .nth(0)
            .expect("should be an integer")
            .parse()
            .unwrap();

        let b: usize = line
            .split_whitespace()
            .nth(1)
            .expect("should be an integer")
            .parse()
            .unwrap();

        graph.add_edge(a,b);
    }
        return graph;
}

pub fn degeneracy(mut graph: Graph) -> (VecDeque<usize>, usize) {

    let mut degeneracy: usize = 0;
    let mut order: VecDeque<usize> = VecDeque::new();
    
    while graph.nnodes() > 0 {
        // finding min node
        let (node_to_elim, min_degree) = graph.find_min_degree_vertex().unwrap();
        //println!("node {:?} degree {:?}", node_to_elim, min_degree);
    
        // updating degeneracy
        if min_degree > degeneracy {
            degeneracy = min_degree;
        }
    
        graph.delete_node(node_to_elim);
        order.push_back(node_to_elim);
    }
    
    return (order, degeneracy);
    
}

pub fn degen_solver(graph: Graph, mut order: VecDeque<usize>, depth: usize) -> Vec<usize> {

    let mut ds: Vec<usize> = Vec::new();

    if graph.nnodes() > 0 {
    
        let mut u = order.pop_front().unwrap();
        while !graph.ngbh.contains_key(&u) {
            u = order.pop_front().unwrap();
        }
        println!("branching over {:?} at depth {:?}", u, depth);

        let mut test_set: Vec<usize> = vec![u];
        for v in graph.neighborhood(u) {
            test_set.push(v);
        }

        let first_round: bool = true;

        for w in test_set {
            let mut graph_w = Graph { ngbh: graph.ngbh.clone() };
            for x in graph.neighborhood(w) {
                graph_w.delete_node(x);
            }
            graph_w.delete_node(w);
            let mut ds_w: Vec<usize> = degen_solver(graph_w, order.clone(), depth+1);
            ds_w.push(w);

            if first_round {
                ds = ds_w;
            }
            else {
                if ds.len() > ds_w.len() {
                    ds = ds_w
                }
            }
        }
    }

    return ds;
}

pub fn max_sat_solver(graph: Graph) -> Vec<bool> {

    let mut solver = rsmaxsat::solver::evalmaxsat::CadicalEvalMaxSATSolver::new();

    let n: usize = graph.ngbh.len();

    for (u, l) in graph.ngbh.iter() {
        let mut clause_vec: Vec<i32> = Vec::new();
        clause_vec.push(*u as i32);
        for v in l {
            clause_vec.push(*v as i32);
        }
        solver.add_clause(&clause_vec, Some((n+1) as i64));

        solver.add_clause(&vec![-(*u as i32)], Some(1));
    }

    solver.solve();

    let mut ret_value: Vec<bool> = Vec::new();
    for u in 1..=n {
        ret_value.push(solver.value(u as i32));
    }

    return ret_value;
}
