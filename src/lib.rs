//use std::path::PathBuf;
use std::io::BufRead;
use std::collections::HashMap;
//use rsmaxsat::solver::MaxSATSolver;
use std::collections::VecDeque;
//use structopt::StructOpt;


//#[derive(Debug, StructOpt)]
//pub struct Opt {
//    /// Input file, using the graph format of the PACE 2021 challenge.
//    /// `stdin` if not specified.
//    #[structopt(parse(from_os_str))]
//    pub input: Option<PathBuf>,
//
//    /// Output file. `stdout` if not specified.
//    #[structopt(parse(from_os_str))]
//    output: Option<PathBuf>,
//
//    /// Seed used for all rng. Unsigned 64bit integer value. Defaults to '0' if missing.
//    #[structopt(short, long)]
//    seed: Option<u64>,
//
//}

#[derive(Debug)]
pub struct Instance {
    graph: Graph,
    color: HashMap<usize, Color>,
    suppressed_solution_vertices: Vec<usize>,
    weighted_clauses: Vec<Clause>,
}

pub fn initialize_instance(input_graph: Graph) -> Instance {
    let mut color_map = HashMap::<usize, Color>::new();
    for u in input_graph.ngbh.keys() {
        color_map.insert(*u, Color::BLUE);
    }
    
    Instance {
        graph: input_graph,
        color: color_map,
        suppressed_solution_vertices: Vec::<usize>::new(),
        weighted_clauses: Vec::<Clause>::new(),
    }
}

#[derive(Debug)]
struct Clause {
    vertices: Vec<usize>,
    values: Vec<usize>, // combinations of values as int.
    size: usize,
}

#[derive(Debug,Clone)]
pub enum Color {
    RED, // is in the dominating set
    WHITE, // is dominated already 
    BLUE // must be dominated or be in dominating set. default state.
}

#[derive(Default,Debug)]
pub struct Graph {
    pub ngbh: HashMap<usize, Vec<usize>>,
}

pub struct TreeDec {
    pub bags: HashMap<usize, Vec<usize>>,
    pub tree_ngbh: HashMap<usize, Vec<usize>>,
}

impl Instance {
    pub fn degree1_rule(&mut self) -> bool {
        let mut deg1_vertices: Vec<(usize,usize)> = Vec::<(usize,usize)>::new();
        for (u, neighborhood) in &self.graph.ngbh {
            if neighborhood.len()==1 {
                deg1_vertices.push((*u,neighborhood[0]));
                break;
            }
        }
        for (u,v) in deg1_vertices {
            self.graph.ngbh.remove(&u);
            self.color.remove(&u);
            for w in &self.graph.ngbh[&v] {
                if *w==u {
                    continue;
                }
                self.color.insert(*w, Color::WHITE);    
            }
            self.color.remove(&v);
            self.graph.ngbh.remove(&v);
            self.suppressed_solution_vertices.push(v);
            return true;
        }
        return false;
    }
}

impl Graph {
    fn add_node(&mut self, u: usize) {
        self.ngbh.entry(u).or_insert_with(Vec::new);
//        self.color.entry(u).or_insert(Color::BLUE);
    }

    fn add_edge(&mut self, u: usize, v: usize) {
        self.ngbh.entry(u).or_insert_with(Vec::new).push(v);
        self.ngbh.entry(v).or_insert_with(Vec::new).push(u);
    }

    pub fn nnodes(&self) -> usize {
        return self.ngbh.len();
    }

    fn find_min_degree_vertex(&self) -> Option<(usize,usize)> {
        self.ngbh
            .iter()
            .min_by_key(|(_, vec)| vec.len())
            .map(|(&key, vec)| (key, vec.len()))
    }

    fn find_max_degree_vertex(&self) -> Option<(usize,usize)> {
        self.ngbh
            .iter()
            .max_by_key(|(_, vec)| vec.len())
            .map(|(&key, vec)| (key, vec.len()))
    }

    fn delete_node(&mut self, node_to_delete: usize) {
        for neighbor in self.neighborhood(node_to_delete) {
            let vec = self.ngbh.get_mut(&neighbor).unwrap();
            vec.retain(|&x| x != node_to_delete);
        }
        self.ngbh.remove_entry(&node_to_delete);
    }

    fn neighborhood(&self, u: usize) -> Vec<usize> {
        self.ngbh.get(&u).unwrap().to_vec()
    }
    
    pub fn degree(&self, u: usize) -> usize {
        self.neighborhood(u).len()
    }

//    pub fn degree1_rule(&mut self, u: usize) {
//        let v = self.ngbh.get(&u).unwrap().to_vec().pop().unwrap();
//        self.color.insert(v, Color::RED);
//        self.delete_node(u);
//        for w in self.neighborhood(v) {
//            self.color.insert(w, Color::WHITE);
//        }
//    }

    fn tw_eliminate(&mut self, u: usize) {
        let ngbh_u = self.ngbh.get(&u).unwrap().clone();
        for x in &ngbh_u {
            for y in &ngbh_u {
                if x != y {
                    self.add_edge(*x, *y);
                }
            }
        }
        self.delete_node(u);
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

//pub fn degen_solver(graph: Graph, mut order: VecDeque<usize>, depth: usize) -> Vec<usize> {
//
//    let mut ds: Vec<usize> = Vec::new();
//
//    if graph.nnodes() > 0 {
//    
//        let mut u = order.pop_front().unwrap();
//        while !graph.ngbh.contains_key(&u) {
//            u = order.pop_front().unwrap();
//        }
//        println!("branching over {:?} at depth {:?}", u, depth);
//
//        let mut test_set: Vec<usize> = vec![u];
//        for v in graph.neighborhood(u) {
//            test_set.push(v);
//        }
//
//        let first_round: bool = true;
//
//        for w in test_set {
//            let mut graph_w = Graph { ngbh: graph.ngbh.clone(), color: graph.color.clone() };
//            for x in graph.neighborhood(w) {
//                graph_w.delete_node(x);
//            }
//            graph_w.delete_node(w);
//            let mut ds_w: Vec<usize> = degen_solver(graph_w, order.clone(), depth+1);
//            ds_w.push(w);
//
//            if first_round {
//                ds = ds_w;
//            }
//            else {
//                if ds.len() > ds_w.len() {
//                    ds = ds_w
//                }
//            }
//        }
//    }
//
//    return ds;
//}

//pub fn max_sat_solver(graph: Graph) -> Vec<bool> {
//
//    let mut solver = rsmaxsat::solver::evalmaxsat::CadicalEvalMaxSATSolver::new();
//
//    let n: usize = graph.ngbh.len();
//
//    for (u, l) in graph.ngbh.iter() {
//        let mut clause_vec: Vec<i32> = Vec::new();
//        clause_vec.push(*u as i32);
//        for v in l {
//            clause_vec.push(*v as i32);
//        }
//        solver.add_clause(&clause_vec, Some((n+1) as i64));
//
//        solver.add_clause(&vec![-(*u as i32)], Some(1));
//    }
//
//    solver.solve();
//
//    let mut ret_value: Vec<bool> = Vec::new();
//    for u in 1..=n {
//        ret_value.push(solver.value(u as i32));
//    }
//
//    return ret_value;
//}

//pub fn greedy_tree_dec(input_graph: &Graph) -> TreeDec {
//    let mut graph = Graph {
//        nnodes: input_graph.nnodes,
//        edges: input_graph.edges.clone(),
//        adj_list: input_graph.adj_list.clone(),
//    };
//
//    let bags: HashMap<usize, Vec<usize>> = HashMap::new();
//    let tree_adj = HashMap<usize, Vec<usize>> = HashMap::new();
//
//    // first: finding order
//    let order: Vec<usize> = Vec::new();
//
//    let mut ub = 0;
//    while graph.nnodes > 0 {
//        if let Some(v) = graph
//            .vertices()
//            .into_iter()
//            .min_by(|v, u| graph.degree(*v).cmp(&graph.degree(*u)))
//        {
//            ub = max(ub, graph.degree(v));
//
//            graph.tw_eliminate(v);
//            order.push(v);
//        }
//    }
//
//    let mut graph2 = Graph {
//        nnodes: input_graph.nnodes,
//        edges: input_graph.edges.clone(),
//        adj_list: input_graph.adj_list.clone(),
//    };
//
//    for v in order {
//
//    }
//
//    return ub;
//}
