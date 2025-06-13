//use std::convert::TryFrom;
//use std::fs::{File, OpenOptions};
//use std::io;
//use std::io::{stdin, stdout, BufReader};
//use std::path::PathBuf;

use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

use pace_2025_solver::*;

fn main() {
    let graph: Graph = parse_graph();

    let mut instance: Instance = initialize_instance(graph);

    // applying degree1 rule
    loop {
        let b: bool = instance.degree1_rule();
        if !b {
            break;
        }
    }

    let mut solution = treewidth_solver_own_graph_format(&instance);

    println!("{:?}", solution.len());

    for u in &instance.suppressed_solution_vertices {
        solution.push(*u);
    }

    for u in solution {
        println!("{:?}", u);
    }

//    let is_ds = check_is_ds(&solution, &copy_graph);
//    assert!(is_ds);
//    println!("is solution ds {:?}", is_ds);

}

