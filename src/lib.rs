use std::collections::HashMap;
use std::io::BufRead;
use rsmaxsat::solver::MaxSATSolver;

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
