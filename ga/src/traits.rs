use crate::population::MutationConfig;

pub trait Crossover {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self;
}

pub trait Fitness {
    fn calculate_fitness(&mut self, seed: [u8; 32]) -> Option<f64>;
}

pub trait FitnessRetrieve {
    fn get_fitness(&self) -> Option<f64>;
}

pub trait Mutate {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self;
}

pub trait Generate {
    fn generate(seed: [u8; 32]) -> Self;
}
