use crate::genome::{Genome, Fitness, Population, Mutate};
use rand::Rng;

mod genome;

fn main() {
    println!("Hello, world!");

    let g1 = MyGenome { data: vec![1, 2, 3, 4, 5], ..Default::default()};
    let g2 = MyGenome { data: vec![20, 21, 22, 23, 24, 26, 29], ..Default::default()};
    let g3 = MyGenome { data: vec![20, 21, 22, 23, 24, 26, 29], ..Default::default()};

    let mut g4 = g1.mutate();
    let mut g5 = g1.crossover(&g2);
    let mut g6 = g1.crossover(&g2);
    g4.calculate_fitness();
    g5.calculate_fitness();
    g6.calculate_fitness();
    println!("{:?}", g4);
    println!("{:?}", g5);
    println!("{:?}", g6);

    let mut p: Population<MyGenome> = Population::new();
    println!("{:?} with {:?}", p, p.members);
    p.tick();
    println!("{:?}", p);
}

const MAX_LEN: usize = 10;

const MAX_VALUE: i64 = 256;
const MIN_VALUE: i64 = -256;

const ALLELE_MUT_CHANCE: i32 = 20;

type MyGenome = Genome<i64>;

impl genome::Fitness for Genome<i64> {
    fn calculate_fitness(&mut self) -> Option<i64> {
        if self.fitness == None {
            self.fitness = Some(self.data.iter().sum());
        }
        self.fitness
    }
}

impl genome::Mutate for MyGenome {
    fn mutate(&self) -> MyGenome {
        let mut rng = rand::thread_rng();
        let mut new_data = self.data.clone();
        for (_, v) in new_data.iter_mut().enumerate() {
            if rng.gen::<i32>() % 100 < ALLELE_MUT_CHANCE {
                // This is a hard mutate, where a big change is made
                // We could also add / subtract a small value instead
                *v = rng.gen_range(MIN_VALUE..MAX_VALUE);
            }
        }
        MyGenome { data: new_data, ..Default::default() }
    }
}

impl genome::Generate for MyGenome {
    fn generate() -> MyGenome {
        let mut rng = rand::thread_rng();
        let mut data: Vec<i64> = Vec::new();
        for _ in [1..=rng.gen_range(1..=MAX_LEN)] {
            data.push(rng.gen_range(MIN_VALUE..MAX_VALUE));
        }
        MyGenome { data: data, ..Default::default() }
    }
}
