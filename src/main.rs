use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{stdin, stdout, BufReader};
use std::path::PathBuf;
use pace_2025_solver::*;

fn main() {
//    let opt = Opt::from_args();
//
//    let instance: Instance  = match opt.input {
//        Some(path) => {
//            let file = File::open(path)?;
//            let reader = PaceReader(BufReader::new(file));
//            Instance::try_from(reader)?
//        }
//        None => {
//            let stdin = stdin();
//            let reader = PaceReader(stdin.lock());
//            Instance::try_from(reader)?
//        }
//    };
//
//    let file = match opt.output {
//        Some(path) => Some(OpenOptions::new().write(true).create(true).open(path)?),
//        None => None,
//    };
//
    let graph: Graph = parse_graph();
    println!("{:?}", graph.ngbh);

    let mut instance: Instance = initialize_instance(graph);
    println!("{:?}", instance);

    // applying degree1 rule
    loop {
        let b: bool = instance.degree1_rule();
        if !b {
            break;
        }
    }

    println!("{:?}", instance);

//    let td = heuristic_td(instance.graph);
//
//    instance.reduce_protrusions(td);
//
//    instance.sat_solver();

//    match file {
//        Some(file) => PaceWriter::new(&td, &graph, file).output(),
//        None => {
//            let writer = stdout();
//            PaceWriter::new(&td, &graph, writer).output()
//        }
//    }
}

