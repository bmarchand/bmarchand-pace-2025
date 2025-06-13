use std::io::BufRead;
use std::collections::HashMap;
//use rsmaxsat::solver::MaxSATSolver;
use std::collections::VecDeque;
use std::cmp::min;
use arboretum_td::SafeSeparatorLimits;
use arboretum_td::solver::Solver;
use arboretum_td::graph::MutableGraph;
//use arboretum_td::graph::BaseGraph;
use arboretum_td::graph::HashMapGraph;
//use arboretum_td::tree_decomposition::TreeDecomposition;


#[derive(Debug)]
pub struct Instance {
    pub graph: Graph,
    pub suppressed_solution_vertices: Vec<usize>,
    pub already_dominated: Vec<usize>,
    //color: HashMap<usize, Color>,
    //weighted_clauses: Vec<Clause>,
}

pub fn initialize_instance(input_graph: Graph) -> Instance {
//    let mut color_map = HashMap::<usize, Color>::new();
//    for u in input_graph.ngbh.keys() {
//        color_map.insert(*u, Color::BLUE);
//    }
    Instance {
        graph: input_graph,
        suppressed_solution_vertices: Vec::<usize>::new(),
        already_dominated: Vec::<usize>::new(),
    }
}

//#[derive(Debug)]
//struct Clause {
//    vertices: Vec<usize>,
//    values: Vec<usize>, // combinations of values as int.
//    size: usize,
//}

#[derive(Debug,Clone,PartialEq)]
pub enum Color {
    RED, // is in the dominating set
    WHITE, // is dominated already 
    BLUE // must be dominated or be in dominating set. default state.
}


#[derive(Debug,Clone,PartialEq,Copy)]
pub enum DPColor {
    BLACK, // is in the dominating set
    WHITE, // not in dominating set, must be dominated 
    GREY, // not in dominated set, but already dominated.
}

#[derive(Default,Debug)]
pub struct Graph {
    pub ngbh: HashMap<usize, Vec<usize>>,
}

#[derive(Debug,Clone,Copy,PartialEq)]
pub enum BagType {
    Join,
    IntroduceNode,
    IntroduceEdge,
    ForgetNode,
    Unset,
}

impl Default for BagType {
    fn default() -> Self {
        BagType::Unset
    }
}

#[derive(Default,Debug)]
pub struct Bag {
    pub content: Vec<usize>,
    pub id: usize,
    pub bag_type: BagType,
//    pub vertices: (usize, usize), 
}

#[derive(Default,Debug)]
pub struct TreeDec {
    pub bags: HashMap<usize,Bag>,
    pub tree_ngbh: HashMap<usize,Vec<usize>>,
    pub children: HashMap<usize,Vec<usize>>,
}

impl TreeDec {

//    fn remove_vertex(&mut self, u: usize) {
//        let mut bags_with_u: Vec<usize> = Vec::new();
//        for (id, bag) in &self.bags {
//            if bag.content.contains(&u) {
//                bags_with_u.push(*id);
//            }
//        }
//        for id in bags_with_u {
//            let mut new_content: Vec<usize> = Vec::new();
//            let bag = &self.bags[&id];
//            for v in &bag.content {
//                if *v != u {
//                    new_content.push(*v);
//                }
//            }
//            self.bags.insert(id, Bag { content : new_content, id : id, bag_type : bag.bag_type});
//        }
//    }
    fn bag_dp_order(&self) -> Vec<usize> {
        let root: usize = self.smallest_bag_id();

        let mut rev_order : Vec<usize> = Vec::new();
        let mut queue : Vec<usize> = Vec::new();
        queue.push(root);

        while queue.len() > 0 {
            let u: usize = queue.pop().unwrap();

            rev_order.push(u);

            for v in &self.children[&u] {
                queue.push(*v);
            }
        }

        let mut order: Vec<usize> = Vec::new();

        while rev_order.len() > 0 {
            let u: usize = rev_order.pop().unwrap();
            order.push(u);
        }

        return order;
    }

    fn smallest_bag_id(&self) -> usize {
        let mut min_id : usize = usize::MAX;
        for u in self.bags.keys() {
            if *u < min_id {
                min_id = *u;
            }
        }
        return min_id;
    }

    fn largest_bag_id(&self) -> usize {
        let mut max_id : usize = 0;
        for u in self.bags.keys() {
            if *u > max_id {
                max_id = *u;
            }
        }
        return max_id;
    }

    fn pick_root(&mut self) {
        let root_id = self.smallest_bag_id();

        let mut queue: Vec<(usize,usize)> = vec![(usize::MAX,root_id)];

        while queue.len() > 0 {
            let (u,v) = queue.pop().unwrap();

            let mut new_children: Vec<usize> = Vec::new();
            for w in &self.tree_ngbh[&v] {
                if *w != u {
                    new_children.push(*w);
                    queue.push((v,*w));
                }
            }
            self.children.insert(v,new_children);
        }
    }

    fn make_root_empty(&mut self) {
        let root_id = self.smallest_bag_id();
        let max_id = self.largest_bag_id();

        if self.bags[&root_id].content.len() > 0 {
            // root not empty
            let old_content: Vec<usize> = self.bags[&root_id].content.clone();
            let u = old_content.iter().min().unwrap();
            let mut new_content: Vec<usize> = Vec::new();
            for v in &old_content {
                if *v != *u {
                    new_content.push(*v);
                }
            }
            let new_id: usize = max_id + 1;

            //actual changing
            let new_root_bag : Bag = Bag {
                content : new_content,
                id : root_id,
                bag_type : BagType::ForgetNode,
                //vertices : (*u, usize::MAX), // forgetting u
            };
            let new_bag: Bag = Bag {
                content : old_content,
                id : new_id,
                bag_type : BagType::Unset,
                //vertices : (usize::MAX,usize::MAX), // don't know yet
            };
            self.bags.insert(root_id, new_root_bag);

            let old_children = self.children[&root_id].clone();
            self.children.insert(root_id, vec![max_id+1]);
            self.children.insert(max_id+1, old_children);
            self.bags.insert(new_id, new_bag);
            self.make_root_empty();
        }
    }

    fn make_nice(&mut self) {
        self.pick_root();
        self.make_root_empty();

        let root_id = self.smallest_bag_id();

        let mut queue: Vec<usize> = vec![root_id];


        while queue.len() > 0 {
            let v = queue.pop().unwrap();
            let max_id = self.largest_bag_id();
            if self.bags[&v].bag_type==BagType::Unset {
                if self.children[&v].len() >= 2 {
                    // select `first child'
                    let first_child: usize = self.children[&v][0];

                    let new_bag1 : Bag = Bag {
                        content : self.bags[&v].content.clone(),
                        id : v,
                        bag_type : BagType::Join,
                        ..Default::default()
                    };

                    let new_bag2 : Bag = Bag {
                        content : self.bags[&v].content.clone(),
                        id : max_id+1,
                        bag_type : BagType::Unset,
                        ..Default::default()
                    };
                    let new_bag3 : Bag = Bag {
                        content : self.bags[&v].content.clone(),
                        id : max_id+2,
                        bag_type : BagType::Unset,
                        ..Default::default()
                    };
                    self.bags.insert(new_bag1.id, new_bag1);
                    self.bags.insert(new_bag2.id, new_bag2);
                    self.bags.insert(new_bag3.id, new_bag3);
                    let old_children : &Vec<usize> = &self.children[&v];
                    let mut new_children : Vec<usize> = Vec::new();
                    for w in old_children {
                        if *w !=first_child {
                            new_children.push(*w);
                        }
                    }
                    self.children.insert(v, vec![max_id+1,max_id+2]);
                    self.children.insert(max_id+1, vec![first_child]);
                    self.children.insert(max_id+2, new_children);
                }
                if self.children[&v].len() == 1 {
                    // select `child'
                    let child: usize = self.children[&v][0];
                    
                    let mut forgotten_vertices: Vec<usize> = Vec::new();
                    for x in &self.bags[&child].content {
                        if !self.bags[&v].content.contains(x) {
                            forgotten_vertices.push(*x);
                        }
                    }

                    if forgotten_vertices.len() > 1 {
                        let picked_vertex: usize = forgotten_vertices.pop().unwrap();
            
                        let max_id = self.largest_bag_id();
                        let mut new_content: Vec<usize> = Vec::new();
                        for x in &self.bags[&child].content {
                            if self.bags[&v].content.contains(x) {
                                new_content.push(*x);
                            }
                        }
                        new_content.push(picked_vertex);
                       
                        let new_bag : Bag = Bag {
                            content : new_content,
                            id : max_id + 1,
                            ..Default::default()
                        };

                        self.bags.insert(v, Bag {
                            content : self.bags[&v].content.clone(),
                            id : v,
                            bag_type : BagType::ForgetNode,
                            ..Default::default()
                        });

                        self.bags.insert(max_id+1, new_bag);
                        self.children.insert(v, vec![max_id+1]);
                        self.children.insert(max_id+1, vec![child]);

                    }
                    
                }
            }
            for w in &self.children[&v] {
                queue.push(*w);
            }
        }

    }
}

impl Instance {
    pub fn degree1_rule(&mut self) -> bool {
        let mut deg1_vertices: Vec<(usize,usize)> = Vec::<(usize,usize)>::new();
        for (u, neighborhood) in &self.graph.ngbh {
            if neighborhood.len()==1 && !self.already_dominated.contains(u) {
                deg1_vertices.push((*u,neighborhood[0]));
                break;
            }
        }
        assert!(deg1_vertices.len()<=1);
        for (u,v) in deg1_vertices {
            self.graph.ngbh.remove(&u);
            //self.color.remove(&u);
            let mut new_ngbhs: Vec<(usize,Vec<usize>)> = Vec::new();
            for w in &self.graph.ngbh[&v] {
                if *w==u {
                    continue;
                }
                self.already_dominated.push(*w);
                //self.color.insert(*w, Color::WHITE);    
                let mut new_ngbh : Vec<usize> = Vec::new();
                for x in &self.graph.ngbh[&w] {
                    if *x==v {
                        continue;
                    }
                    new_ngbh.push(*x);
                }
                new_ngbhs.push((*w, new_ngbh));
            }

            for (w, new_ngbh) in new_ngbhs {
                self.graph.ngbh.insert(w, new_ngbh);
            }

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
    //fn find_vertex_to_eliminate(&self) -> (usize,usize) {
    //    let mut min_deg = usize::MAX;
    //    let mut min_deg_vertex = 0; //dummy value
    //    for (v, neighbors) in &self.ngbh {
    //        if neighbors.len()<8 {
    //            return (*v,neighbors.len())
    //        }
    //        if neighbors.len() < min_deg {
    //            min_deg_vertex = *v;
    //            min_deg = min(min_deg, neighbors.len());
    //        }
    //    }
    //    return (min_deg_vertex, min_deg);
    //}

    fn find_min_degree_vertex(&self) -> (usize,usize) {
        let mut min_deg = usize::MAX;
        let mut min_deg_vertex = 0; //dummy value
        for (v, neighbors) in &self.ngbh {
            if neighbors.len() < min_deg {
                min_deg_vertex = *v;
                min_deg = min(min_deg, neighbors.len());
            }
        }
        return (min_deg_vertex, min_deg);
    }

    //fn find_max_degree_vertex(&self) -> Option<(usize,usize)> {
    //    self.ngbh
    //        .iter()
    //        .max_by_key(|(_, vec)| vec.len())
    //        .map(|(&key, vec)| (key, vec.len()))
    //}

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

//    fn tw_eliminate(&mut self, u: usize) {
//        let ngbh_u = self.ngbh.get(&u).unwrap().clone();
//        for x in &ngbh_u {
//            for y in &ngbh_u {
//                if x != y {
//                    self.add_edge(*x, *y);
//                }
//            }
//        }
//        self.delete_node(u);
//    }
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
        let (node_to_elim, min_degree) = graph.find_min_degree_vertex();
    
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

pub fn heuristic_td(input_graph: &Graph) -> usize {

    let mut graph: HashMapGraph = HashMapGraph::new();
    
    for (u,ngbhs) in &input_graph.ngbh {
        graph.add_vertex(*u);
        for v in ngbhs {
            graph.add_edge(*u,*v);
        }
    }


    let components = graph.connected_components();

    let mut width: usize = 0;

    for sub_graph in components.iter().map(|c| graph.vertex_induced(c)) {
//        let td = Solver::auto(&sub_graph).solve(&sub_graph);
        let td = Solver::default_heuristic()
                        .safe_separator_limits(
                            SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                        )
                        .seed(Some(0))
                        .solve(&sub_graph);
        if td.max_bag_size-1 > width {
            width = td.max_bag_size - 1;
        }
    }

    return width;
}

//fn protrusion_detection_connected_graph(graph: &Graph) -> Vec<(TreeDec, Vec<usize>)> {
//    
//    let width_limit: usize = 15;
//
//    let mut low_degree_vertices : Vec<usize> = Vec::new();
//
//    for (u, neighborhood) in graph.ngbh {
//        if neighborhood.len() < width_limit {
//            low_degree_vertices.push(u);
//        }
//    }
//
//    // while there are low degree vertices
//    while low_degree_vertices.len() > 0 {
//        
//    }
//}

fn tree_dec_connected_graph(graph: &HashMapGraph) -> TreeDec {
    
    let td = Solver::default_heuristic()
                        .safe_separator_limits(
                            SafeSeparatorLimits::default().use_min_degree_for_minor_safe(true),
                        )
                        .seed(Some(0))
                        .solve(&graph);

    let mut tree_dec = TreeDec::default(); 

    for bag in td.bags {
        let mut content : Vec<usize> = Vec::new();
        let mut ngbh: Vec<usize> = Vec::new();
        for u in bag.vertex_set {
            content.push(u);
        }
        content.sort();
        let new_bag: Bag = Bag {
            content: content,
            id: bag.id,
            bag_type: BagType::Unset,
            //vertices: (usize::MAX, usize::MAX), 
        };

        tree_dec.bags.insert(bag.id, new_bag);
        for id2 in bag.neighbors {
            ngbh.push(id2);
        }
        tree_dec.tree_ngbh.insert(bag.id, ngbh);
    }

    return tree_dec;
}

pub fn treewidth_solver_own_graph_format(instance: &Instance) -> Vec<usize> {

    let mut graph: HashMapGraph = HashMapGraph::new();

    for (u,neighbors) in &instance.graph.ngbh {
        for v in neighbors {
            graph.add_edge(*u,*v);
        }
    }

    let components = graph.connected_components();

    let mut sol: Vec<usize> = Vec::new();
    for sub_graph in components.iter().map(|c| graph.vertex_induced(c)) {
        for u in treewidth_solver_connected_graph(&sub_graph, &instance) {
            sol.push(u);
        }
    }

    return sol;
}


fn color2int(constraints: &Vec<DPColor>) -> usize {
    let mut index : u32 = 0;
    for (k,c) in constraints.iter().enumerate() {
        if *c==DPColor::BLACK {
            index += (3 as u32).pow(k as u32);
        }
        if *c==DPColor::GREY {
            index += 2*(3 as u32).pow(k as u32);
        }
    }
//    for (k,c) in constraints.iter().enumerate() {
//        if *c==DPColor::BLACK {
//            index += 1 << 2*k+1;
//            index += 1 << 2*k;
//        }
//        if *c==DPColor::GREY {
//            index += 1 << 2*k;
//        }
//    }
//
    return index as usize;
}

fn color(u: usize, constraints: &Vec<DPColor>, bag_content: &Vec<usize>) -> DPColor {
    for (k,v) in bag_content.iter().enumerate() {
        if *v==u {
            let c: DPColor = constraints[k];
            return c;
        }
    }
    assert!(false);
    return DPColor::WHITE;
}


fn backtrace(bag_id: usize, 
    constraints: &Vec<DPColor>,
    tree_dec: &TreeDec,
    solution_builder: &HashMap<(usize,usize), usize>, 
    solution_pointer: &HashMap<(usize,usize), Vec<DPColor>>, 
    join_solution: &HashMap<(usize,usize), (Vec<DPColor>,Vec<DPColor>)>) -> Vec<usize> {

    let mut return_value: Vec<usize> = Vec::new();
    
    let index: usize = color2int(constraints);
    
    let bag_type: BagType = tree_dec.bags[&bag_id].bag_type;

    if bag_type==BagType::Join {
        let (coloring1, coloring2) = join_solution.get(&(bag_id,index)).unwrap();
        let child1: usize = tree_dec.children[&bag_id][0];
        let child2: usize = tree_dec.children[&bag_id][1];
        let sub_sol1: Vec<usize> = backtrace(child1, coloring1, tree_dec, solution_builder, solution_pointer, join_solution);
        for u in sub_sol1 {
            return_value.push(u);
        }
        let sub_sol2: Vec<usize> = backtrace(child2, coloring2, tree_dec, solution_builder, solution_pointer, join_solution);
        for u in sub_sol2 {
            return_value.push(u);
        }
    }
    if tree_dec.children[&bag_id].len()==1 {
        let child = tree_dec.children[&bag_id][0];
        let coloring = solution_pointer.get(&(bag_id,index)).unwrap();
        if let Some(u) = solution_builder.get(&(bag_id,index)) {
            return_value.push(*u);
        }
        let sub_sol: Vec<usize> = backtrace(child, coloring, tree_dec, solution_builder, solution_pointer, join_solution);
        for u in sub_sol {
            return_value.push(u);
        }
    }

    return return_value;
}

pub fn check_is_ds(solution: &Vec<usize>, graph: &Graph) -> bool {
    for (vertex, neighborhood) in &graph.ngbh {
        if solution.contains(&vertex) {
            continue;
        }
        let mut dominated = false;
        for x in neighborhood {
            if solution.contains(&x) {
                dominated = true;
                break;
            }
        }
        if !dominated {
            return false;
        }
    }
    return true;
}

//fn size_as_subset(x : usize, nbits : usize) -> usize {
//    let mut size : usize = 0;
//    for k in 0..nbits {
//        let is_k_one : bool = ((x >> k) & 1) != 0;
//        if is_k_one {
//            size += 1;
//        }
//    }
//    return size;
//}

///// Given two vectors v1,v2 of size 2^n, whose values
///// are within 0 and M, it returns the vector W of size 2^n
///// whose values are 
///// W[S] = min_{A,B such that A\cup B = S and A\cap B=\empty} V1[A]+v2[B]
//fn subset_convolution(v1: Vec<usize>, v2 : Vec<usize>, n: usize) -> Vec<usize> {
//
//    // finding M
//    let mut M :usize = 0;
//    for k in 0..v1.len() {
//        if v1[k] > M {
//            M = v1[k];
//        }
//        if v2[k] > M {
//            M = v2[k];
//        }
//    }
//
//    // the base of the exponential
//    let beta : u32 = v1.len() as u32 + 1u32;
//
//    let mut big_int1 : usize = 0;
//    let mut big_int2 : usize = 0;
//
//    // the exponentiated values
//    for k in 0..v1.len() {
//        big_int1 += beta.pow(v1[k]);
//        big_int2 += beta.pow(v2[k]);
//    }
//    let magic_result = big_int1 * big_int2;
//
//    let w : Vec<usize> = Vec::new();
//    
//    for xu in 1..(1 << n) {
//        
//    }
//}
//
//fn subset_convolution_join_bags_fill(
//    bag_id: usize, 
//    tree_dec: &TreeDec,
//    dp_table: &mut HashMap<(usize,usize), usize>, 
//    join_solution: &mut HashMap<(usize,usize), (Vec<DPColor>,Vec<DPColor>)>)  {
//    
//    let bag_type: BagType = tree_dec.bags[&bag_id].bag_type;
//    let content: Vec<usize> = tree_dec.bags[&bag_id].content;
//
//    if bag_type==BagType::Join {
//        assert!(tree_dec.children[&bag_id].len()==2);
//        
//        let child1: usize = tree_dec.children[&bag_id][0];
//        let child2: usize = tree_dec.children[&bag_id][1];
//
//        for xr in 0..(1 << content.len()) {
//            let size_r : usize = size_as_subset(xr, content.len());
//            let v1: Vec<usize> = Vec::new();
//            let v2: Vec<usize> = Vec::new();
//            for xu in 0..(1 << (content.len()-size_r)) {
//                let mut constraints : Vec<DPColor> = Vec::new();
//                let mut count_black : usize = 0;
//                for k in 0..content.len() {
//                    let is_black : bool = (xr >> k) & 1;
//
//                    if (is_black) {
//                        count_black += 1;
//                    }
//                    let is_white : bool = (xu >> (k-count_black)) & 1 & !is_black;
//
//                    if is_black {
//                        constraints.push(DPColor::BLACK);
//                    }
//                    else if is_white {
//                        constraints.push(DPColor::WHITE);
//                    }
//                    else {
//                        constraints.push(DPColor::GREY);
//                        // not black nor white -> grey
//                    }
//                }
//
//                let value1 : usize = dp_solve(child1, solution_builder, solution_pointer, join_solution, constraints, instance);
//                let value2 : usize = dp_solve(child2, solution_builder, solution_pointer, join_solution, constraints, instance);
//                v1.push(value1);
//                v2.push(value2);
//            }
//
//            let w : Vec<usize> = subset_convolution(v1,v2,content.len()-size_r);
//
//            for xu in 0..(1 << size_r) {
//                //dp_table.insert(...,w[~xu]-size_r);
//            }
//        }
//
//    }
//    else {
//        for child in tree_dec.children[&bag_id] {
//            subset_convolution_join_bags_fill(child, tree_dec, dp_table, join_solution);
//        }
//    }
//}
//

/// Computes a tree decomposition and runs dynamic programming on it to Dominating
/// Set.
///
/// It does not run any-preprocessing rules (such as the degree-1 vertex rule).
/// It does however accept as input an instance that has already been pre-processed,
/// i.e. in which we know that some vertices are already dominated. For these
/// vertices, the DP does not explore the state "WHITE" which means 'must be
/// dominated'.
fn treewidth_solver_connected_graph(graph: &HashMapGraph, instance: &Instance) -> Vec<usize> {


    let mut tree_dec = tree_dec_connected_graph(&graph);
    tree_dec.make_nice();

    let mut dp_table: HashMap<usize, Vec<usize>> = HashMap::new(); 
    let mut solution_builder: HashMap<(usize,usize), usize> = HashMap::new();
    let mut solution_pointer: HashMap<(usize,usize), Vec<DPColor>> = HashMap::new();
    let mut join_solution: HashMap<(usize,usize), (Vec<DPColor>,Vec<DPColor>)> = HashMap::new();
    let empty_constraints: Vec<DPColor> = Vec::new();
    let root: usize = tree_dec.smallest_bag_id();

    //let _opt: usize = dp_solve(root, &tree_dec, &mut dp_table, &mut solution_builder, &mut solution_pointer, &mut join_solution, &empty_constraints, &instance);
    let _opt: usize = iter_dp_solve(&tree_dec, &mut dp_table, &mut solution_builder, &mut solution_pointer, &mut join_solution, &instance);

    let sol: Vec<usize> = backtrace(root, &empty_constraints, &tree_dec, &solution_builder, &solution_pointer, &join_solution);

    return sol;
}

fn iter_dp_solve(tree_dec: &TreeDec,
    dp_table: &mut HashMap<usize,Vec<usize>>, 
    solution_builder: &mut HashMap<(usize,usize), usize>, 
    solution_pointer: &mut HashMap<(usize,usize), Vec<DPColor>>, 
    join_solution: &mut HashMap<(usize,usize), (Vec<DPColor>,Vec<DPColor>)>, 
    instance: &Instance) -> usize {

    for bag_id in tree_dec.bag_dp_order() {

        let bag_size = tree_dec.bags[&bag_id].content.len();
        let bag_content = &tree_dec.bags[&bag_id].content;

        for child in &tree_dec.children[&bag_id] {
            for grand_child in &tree_dec.children[child] {
                dp_table.remove(grand_child);
            }
        }

        if tree_dec.children[&bag_id].len()==1 {

            // computing the set of introduced and forgotten nodes.
            let child = tree_dec.children[&bag_id][0];
            let child_content: &Vec<usize> = &tree_dec.bags[&child].content;

            let mut introduced_nodes: Vec<usize> = Vec::new();
            for u in bag_content {
                if !child_content.contains(&u) {
                    introduced_nodes.push(*u);
                }
            }

            let mut forgotten_nodes: Vec<usize> = Vec::new();
            for u in child_content {
                if !bag_content.contains(u) {
                    forgotten_nodes.push(*u);
                }
            }
            assert!(forgotten_nodes.len()==1);

            let mut v : Vec<usize> = vec![usize::MAX; (3 as u32).pow(bag_size as u32) as usize];
            let v_child = dp_table.get(&child).unwrap();

            // actual minimization
            for index_u32 in 0..(3 as u32).pow(bag_size as u32) {
                let index: usize = index_u32 as usize;

                let mut answer : usize = usize::MAX;
                let constraints : Vec<DPColor> = index2coloring(index as usize, &bag_content);
                let mut lets_continue : bool = false;

                let mut count_ones = 0;
                let mut new_color: HashMap<usize,DPColor> = HashMap::new();
                // check introduced nodes
                for u in &introduced_nodes {
                    if color(*u,&constraints,&bag_content)==DPColor::BLACK {
                        count_ones += 1;
                        for v in &instance.graph.ngbh[&u] {
                            if bag_content.contains(&v) {
                                if child_content.contains(&v) { // not introduced
                                    if color(*v,&constraints,&bag_content)==DPColor::WHITE {
                                        new_color.insert(*v, DPColor::GREY);
                                    }
                                }
                            }
                        }
                    }

                    if color(*u,&constraints,&bag_content)==DPColor::WHITE {

                        let mut dominated: bool = false;
                        if instance.already_dominated.contains(u) {
                            dominated = true;
                        }
                        for v in &instance.graph.ngbh[&u] {
                            if bag_content.contains(&v) {
                                if color(*v,&constraints,&bag_content)==DPColor::BLACK {
                                    dominated = true;
                                }
                            }
                        }
                        if !dominated {
                            v[index] = usize::MAX;
                            lets_continue = true;
                        }
                    }
                }

                if lets_continue {
                    continue;
                }

                let forgotten_node = forgotten_nodes[0];
                let mut compatible_coloring1: Vec<DPColor> = Vec::new();
                let mut compatible_coloring2: Vec<DPColor> = Vec::new();
                for u in child_content {
                    if let Some(color) = new_color.get(u) {
                        compatible_coloring1.push(*color);
                        compatible_coloring2.push(*color);
                    }
                    else if *u==forgotten_node {
                        compatible_coloring1.push(DPColor::BLACK);
                        compatible_coloring2.push(DPColor::WHITE);
                    }
                    else {
                        compatible_coloring1.push(color(*u,&constraints,&bag_content));
                        compatible_coloring2.push(color(*u,&constraints,&bag_content));
                    }
                }

                let index_child1: usize = color2int(&compatible_coloring1);
                let index_child2: usize = color2int(&compatible_coloring2);


                let mut sub_solsize: usize = usize::MAX;
                let ans1: usize = v_child[index_child1];
                let ans2: usize = v_child[index_child2];
     
                let mut is_v_in_ds: bool = true; 
                if ans1 < sub_solsize {
                    sub_solsize = ans1;
                    solution_pointer.insert((bag_id, index), compatible_coloring1.clone());
                }
                if ans2 < sub_solsize {
                    sub_solsize = ans2;
                    solution_pointer.insert((bag_id, index), compatible_coloring2.clone());
                    is_v_in_ds = false;
                }

                if sub_solsize < usize::MAX {
                    answer = count_ones + sub_solsize;
                }
                if is_v_in_ds {
                    solution_builder.insert((bag_id, index), forgotten_node); 
                }
                
                v[index] = answer;
                    
            }
            dp_table.insert(bag_id, v);
        }
        if tree_dec.children[&bag_id].len()==2 {
    
            // then 2 children, with identical content
            let child1: usize = tree_dec.children[&bag_id][0];
            let child2: usize = tree_dec.children[&bag_id][1];
        
            let size_bag : usize = bag_content.len();

            let mut v : Vec<usize> = vec![usize::MAX; (3 as u32).pow(size_bag as u32) as usize];
            let v_child1 = dp_table.get(&child1).unwrap();
            let v_child2 = dp_table.get(&child2).unwrap();
        
            for x in 0..(1 << 2*size_bag) {
                let mut constraints : Vec<DPColor> = Vec::new();
                let mut constraints1 : Vec<DPColor> = Vec::new();
                let mut constraints2 : Vec<DPColor> = Vec::new();
                let mut num_black : usize = 0;
                for k in 0..size_bag {
        
                    let b1: bool = ((x >> (2*k+1)) & 1) != 0;
                    let b2: bool = ((x >> (2*k)) & 1) != 0;
                    if b1 {
                        if b2 {
                            constraints.push(DPColor::GREY);
                            constraints1.push(DPColor::GREY);
                            constraints2.push(DPColor::GREY);
                        }
                        else {
                            num_black += 1;
                            constraints.push(DPColor::BLACK);
                            constraints1.push(DPColor::BLACK);
                            constraints2.push(DPColor::BLACK);
                        }
                    }
                    else {
                        if b2 {
                            constraints.push(DPColor::WHITE);
                            constraints1.push(DPColor::WHITE);
                            constraints2.push(DPColor::GREY);
        
                        }
                        else {
                            constraints.push(DPColor::WHITE);
                            constraints1.push(DPColor::GREY);
                            constraints2.push(DPColor::WHITE);
                        }
                    }
                }

                let index: usize = color2int(&constraints);
                let index_child1: usize = color2int(&constraints1);
                let index_child2: usize = color2int(&constraints2);
        
                let ans1: usize = v_child1[index_child1];
                let ans2: usize = v_child2[index_child2];

                let new_answer : usize;
                if ans1==usize::MAX || ans2==usize::MAX {
                    new_answer = usize::MAX;
                }
                else {
                    new_answer = ans1 + ans2 - num_black;
                }

                if new_answer < v[index] {
                        v[index] = new_answer;
                        join_solution.insert((bag_id,index), (constraints1, constraints2)); 
                }

            }
            dp_table.insert(bag_id, v);
        }
        if tree_dec.children[&bag_id].len()==0 {

            let mut v : Vec<usize> = vec![usize::MAX; (3 as u32).pow(bag_size as u32) as usize];

            // actual minimization
            for index in 0..(3 as u32).pow(bag_size as u32) {
                let mut lets_continue : bool = false;
                let constraints = index2coloring(index as usize, &bag_content);

                let mut count_ones = 0;
                // check introduced nodes
                for u in bag_content {
                    if color(*u,&constraints,&bag_content)==DPColor::BLACK {
                        count_ones += 1;
                    }

                    if color(*u,&constraints,&bag_content)==DPColor::WHITE {
                        //let mut dominated: bool = instance.already_dominated.contains(u);
                        let mut dominated: bool = false;
                        if instance.already_dominated.contains(u) {
                            dominated = true;
                        }
                        for v in &instance.graph.ngbh[&u] {
                            if bag_content.contains(&v) {
                                if color(*v,&constraints,&bag_content)==DPColor::BLACK {
                                    dominated = true;
                                }
                            }
                        }
                        if !dominated {
                            v[index as usize] = usize::MAX;
                            lets_continue = true;
                        }
                    }
                }
                let answer : usize = count_ones;

                if lets_continue {
                    continue;
                }
                v[index as usize] = answer;
            }
            dp_table.insert(bag_id, v);
        }
    }

    return dp_table.get(&0).unwrap()[0];
}

fn index2coloring(index : usize, content: &Vec<usize>) -> Vec<DPColor> {
    let mut color_vec : Vec<DPColor> = Vec::new();
    for k in 0..content.len() {
        let q = (3 as u32).pow(k as u32);
        let r = index/(q as usize);
        if r%3==0 {
            color_vec.push(DPColor::WHITE);
        }
        if r%3==1 {
            color_vec.push(DPColor::BLACK);
        }
        if r%3==2 {
            color_vec.push(DPColor::GREY);
        }
    }
    return color_vec;
}
