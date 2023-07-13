use rand::{Rng, seq::SliceRandom};

const MAX_VALUE: i64 = 256;
const MIN_VALUE: i64 = -256;
const MAX_LEN: usize = 10;
const POP_SIZE: usize = 10;

const ALLELE_MUT_CHANCE: i32 = 20;

const POP_MUT_CHANCE: i32 = 10;
const POP_CROSS_CHANCE: i32 = 20;

#[derive(Debug, Default)]
pub struct Genome {
    pub data: Vec<i64>,
    pub fitness: Option<i64>,
}

impl Genome {
    pub fn new() -> Genome {
        let mut rng = rand::thread_rng();
        let mut data: Vec<i64> = Vec::new();
        for _ in [1..=rng.gen_range(1..=MAX_LEN)] {
            data.push(rng.gen_range(MIN_VALUE..MAX_VALUE));
        }
        Genome { data: data, ..Default::default() }
    }

    pub fn mutate(&self) -> Genome {
        let mut rng = rand::thread_rng();
        let mut new_data = self.data.clone();
        for (_, v) in new_data.iter_mut().enumerate() {
            if rng.gen::<i32>() % 100 < ALLELE_MUT_CHANCE {
                // This is a hard mutate, where a big change is made
                // We could also add / subtract a small value instead
                *v = rng.gen_range(MIN_VALUE..MAX_VALUE);
            }
        }
        Genome { data: new_data, ..Default::default() }
    }

    pub fn crossover(&self, other: &Genome) -> Genome {
        let mut rng = rand::thread_rng();

        let len_a = self.data.len();
        let len_b = other.data.len();

        let amount = rng.gen_range(1..=len_b);
        let new_slice : Vec<i64> = other.data.choose_multiple(&mut rng, amount).cloned().collect();

        let start_a = rng.gen_range(0..len_a);
        let end_a = rng.gen_range(start_a..len_a);

        let mut new_data: Vec<i64> = Vec::new();
        let mut inserted = false;
        for (i, v) in self.data.iter().enumerate() {
            if i < start_a || i > end_a {
                new_data.push(*v);
            } else {
                if !inserted {
                    for n in new_slice.iter() {
                        new_data.push(*n);
                    }
                    inserted = true;
                }
            }
        }

        Genome { data: new_data, ..Default::default() }
    }
}

pub trait Fitness {
    fn calculate_fitness(&mut self) -> Option<i64>;
}

#[derive(Debug)]
pub struct Population {
    pub members: Vec<Genome>,
}

impl Population {
    pub fn new() -> Population {
        let mut members: Vec<Genome> = Vec::new();
        for _ in 1..=POP_SIZE {
            members.push(Genome::new());
        }
        Population { members: members }
    }

    pub fn tick(&mut self) {
        let new_pop : Vec<Genome> = Vec::new();
        // Select biased towards top performers
        // ..this needs running calculate_fitness on all
        // Then randomly choose whether to crossover between chosen, or mutate
        // Repeat until
        // Replace old population with new one
        self.members = new_pop;
    }
}
