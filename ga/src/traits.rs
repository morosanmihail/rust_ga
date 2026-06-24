use rand::rngs::StdRng;

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

/// Selects indices from a population for reproduction.
///
/// Receives the current fitness values (in member order) and returns the
/// chosen index/indices. Keeping the interface index-based makes the trait
/// dyn-compatible and decouples selection logic from genome type.
pub trait SelectionStrategy: Send + Sync + std::fmt::Debug {
    fn select(&self, fitnesses: &[Option<f64>], rng: &mut StdRng) -> Option<usize>;

    fn select_pair(
        &self,
        fitnesses: &[Option<f64>],
        rng: &mut StdRng,
    ) -> Option<(usize, usize)> {
        let a = self.select(fitnesses, rng)?;
        let b = self.select(fitnesses, rng)?;
        Some((a, b))
    }
}
