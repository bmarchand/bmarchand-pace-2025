use pace_2025_solver::*;
use rsmaxsat::solver::MaxSATSolver;

fn main() {
    let graph: Graph = parse_graph();

    let sat_solution = max_sat_solver(graph); 

    // counting true.
    let mut n: usize = 0;

    for b in &sat_solution {
        if *b {
            n += 1;
        }
    }
    println!("{:?}", n);
    for k in 1..=sat_solution.len() {
        if sat_solution[k-1] {
            println!("{:?}",k);
        }
    }
}
