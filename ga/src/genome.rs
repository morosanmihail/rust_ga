use rand::{Rng, seq::SliceRandom};

const POP_SIZE: usize = 10;


const POP_MUT_CHANCE: i32 = 10;
const POP_CROSS_CHANCE: i32 = 20;

#[derive(Debug, Default)]
pub struct Genome<T: Clone + Default + Copy> {
    pub data: Vec<T>,
    pub fitness: Option<T>,
}

impl<T: Clone + Default + Copy> Genome<T> {
    pub fn crossover(&self, other: &Genome<T>) -> Genome<T> {
        let mut rng = rand::thread_rng();

        let len_a = self.data.len();
        let len_b = other.data.len();

        let amount = rng.gen_range(1..=len_b);
        let new_slice : Vec<T> = other.data.choose_multiple(&mut rng, amount).cloned().collect();

        let start_a = rng.gen_range(0..len_a);
        let end_a = rng.gen_range(start_a..len_a);

        let mut new_data: Vec<T> = Vec::new();
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

pub trait Mutate {
    fn mutate(&self) -> Self;
}

pub trait Generate {
    fn generate() -> Self;
}

#[derive(Debug)]
pub struct Population<T: Generate + Mutate + Fitness> {
    pub members: Vec<T>,
}

impl<T: Generate + Mutate + Fitness> Population<T> {
    pub fn new() -> Population<T> {
        let mut members: Vec<T> = Vec::new();
        for _ in 1..=POP_SIZE {
            // members.push(Genome::new());
            members.push(T::generate());
        }
        Population { members: members }
    }

    pub fn tick(&mut self) {
        let new_pop : Vec<T> = Vec::new();
        // Select biased towards top performers
        // ..this needs running calculate_fitness on all
        // Then randomly choose whether to crossover between chosen, or mutate
        // Repeat until
        // Replace old population with new one
        self.members = new_pop;
    }
}
