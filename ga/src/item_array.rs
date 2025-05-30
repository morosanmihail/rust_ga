use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    population::{Genome, MutationConfig},
    traits::{Crossover, FitnessRetrieve, Generate, Mutate},
};

pub const DEFAULT_MIN_LEN: usize = 20;
pub const DEFAULT_MAX_LEN: usize = 20;

#[derive(Default, Clone, Debug)]
pub struct ItemArray<T: Clone + Default + Mutate> {
    inner: Genome<Vec<T>>,
}

impl<T: Clone + Default + Mutate + Generate> ItemArray<T> {
    pub fn get_data(&self) -> &Vec<T> {
        &self.inner.data
    }
    pub fn set_fitness(&mut self, fitness: Option<f64>) {
        self.inner.fitness = fitness;
    }
    pub fn generate_length(min_length: usize, max_length: usize, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);

        ItemArray {
            inner: Genome {
                data: (0..rng.gen_range(min_length..=max_length))
                    .map(|_| T::generate(rng.gen()))
                    .collect(),
                ..Default::default()
            },
        }
    }
}

impl<T: Clone + Default + Mutate> Mutate for ItemArray<T> {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let new_data = self
            .inner
            .data
            .iter()
            .map(|e| {
                if rng.gen::<i32>() % 100 < ((100.0 * config.gene_mutation_chance) as i32) {
                    e.mutate(config, rng.gen())
                } else {
                    e.clone()
                }
            })
            .collect();
        ItemArray {
            inner: Genome {
                data: new_data,
                ..Default::default()
            },
        }
    }
}

impl<T: Clone + Default + Mutate> Crossover for ItemArray<T> {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let min_length = std::cmp::min(self.inner.data.len(), other.inner.data.len());

        let crossover_point = rng.gen_range(0..=min_length);

        let mut offspring =
            Vec::with_capacity(std::cmp::max(self.inner.data.len(), other.inner.data.len()));

        offspring.extend_from_slice(&self.inner.data[..crossover_point]);
        offspring.extend_from_slice(&other.inner.data[crossover_point..]);

        ItemArray {
            inner: Genome {
                data: offspring,
                ..Default::default()
            },
        }
    }
}

impl<T: Clone + Generate + Default + Mutate> Generate for ItemArray<T> {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);

        ItemArray {
            inner: Genome {
                data: (0..rng.gen_range(DEFAULT_MIN_LEN..=DEFAULT_MAX_LEN))
                    .map(|_| T::generate(rng.gen()))
                    .collect(),
                ..Default::default()
            },
        }
    }
}

impl<T: Copy + Default + Mutate> FitnessRetrieve for ItemArray<T> {
    fn get_fitness(&self) -> Option<f64> {
        self.inner.fitness
    }
}
