use crate::genome::{Genome, Fitness, Population};

mod genome;

fn main() {
    println!("Hello, world!");

    let g1 = Genome { data: vec![1, 2, 3, 4, 5], ..Default::default()};
    let g2: Genome = Genome { data: vec![20, 21, 22, 23, 24, 26, 29], ..Default::default()};
    let g3: Genome = Genome { data: vec![20, 21, 22, 23, 24, 26, 29], ..Default::default()};

    let mut g4 = g1.mutate();
    let mut g5 = g1.crossover(&g2);
    let mut g6 = g1.crossover(&g2);
    g4.calculate_fitness();
    g5.calculate_fitness();
    g6.calculate_fitness();
    println!("{:?}", g4);
    println!("{:?}", g5);
    println!("{:?}", g6);

    let mut p = Population::new();
    println!("{:?} with {:?}", p, p.members);
    p.tick();
    println!("{:?}", p);
}

impl genome::Fitness for Genome {
    fn calculate_fitness(&mut self) -> Option<i64> {
        if self.fitness == None {
            self.fitness = Some(self.data.iter().sum());
        }
        self.fitness
    }
}
