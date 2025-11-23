use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Genome<T: Clone + Default> {
    pub data: T,
    pub fitness: Option<f64>,
}

impl<T: Default + Clone> FitnessRetrieve for Genome<T> {
    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    pub gene_mutation_chance: f64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PopulationConfig<T: Generate + Fitness> {
    pub seed: [u8; 32],
    pub pop_size: usize,
    pub crossover_count: usize,
    pub mutate_count: usize,
    pub elitism_count: usize,

    pub mutation_config: MutationConfig,

    pub preseeded_population: Vec<T>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Population<
    T: Generate + Crossover + Mutate + Fitness + FitnessRetrieve + Default + Send + Sync,
> {
    pub members: Vec<T>,
    pub config: PopulationConfig<T>,
    generation: i64,
    seed: [u8; 32],
}

impl<
        T: Generate + Crossover + Mutate + Fitness + FitnessRetrieve + Default + Clone + Send + Sync,
    > Population<T>
{
    pub fn new(mut config: PopulationConfig<T>) -> Population<T> {
        let mut rng: StdRng = SeedableRng::from_seed(config.seed);
        let mut members: Vec<T> = Vec::new();
        members.append(&mut config.preseeded_population);
        for _ in members.len() + 1..=config.pop_size {
            members.push(T::generate(rng.gen()));
        }
        Population {
            seed: rng.gen(),
            members,
            config,
            generation: 1,
        }
    }

    pub fn sort_members(&mut self) {
        self.members.sort_by(|a, b| {
            b.get_fitness()
                .partial_cmp(&a.get_fitness())
                .unwrap_or(std::cmp::Ordering::Less)
        });
    }

    pub fn get_best_member(&mut self) -> &T {
        self.sort_members();
        &self.members[0]
    }

    pub fn tick_parallel(&mut self) {
        let rng: StdRng = SeedableRng::from_seed(self.seed);

        self.members.par_iter_mut().for_each(|m| {
            let mut rng_clone = rng.clone();
            {
                m.calculate_fitness(rng_clone.gen());
            }
        });

        self.post_tick();
    }

    pub fn tick(&mut self) {
        let mut rng: StdRng = SeedableRng::from_seed(self.seed);

        self.members.iter_mut().for_each(|m| {
            m.calculate_fitness(rng.gen());
        });

        self.post_tick();
    }

    fn post_tick(&mut self) {
        let mut new_pop: Vec<T> = Vec::new();
        let mut rng: StdRng = SeedableRng::from_seed(self.seed);

        self.sort_members();

        // Elitism first
        new_pop.extend(
            self.members
                .iter()
                .take(self.config.elitism_count)
                .cloned()
                .collect::<Vec<_>>(),
        );

        // Then mutation
        (0..self.config.mutate_count).for_each(|_| {
            let mutatable_member = self.members.choose(&mut rng);
            if let Some(t) = mutatable_member {
                let mut m = t.mutate(&self.config.mutation_config, rng.gen());
                m.calculate_fitness(rng.gen());
                new_pop.push(m);
            }
        });

        // Then crossover
        (0..self.config.crossover_count).for_each(|_| {
            let crossoverable_members: Vec<&T> =
                self.members.choose_multiple(&mut rng, 2).collect();
            let mut crossoverd_member =
                crossoverable_members[0].crossover(crossoverable_members[1], rng.gen());
            crossoverd_member.calculate_fitness(rng.gen());
            new_pop.push(crossoverd_member);
        });

        // Then newly generated ones
        (new_pop.len()..self.config.pop_size).for_each(|_| {
            let mut generated_member = T::generate(rng.gen());
            generated_member.calculate_fitness(rng.gen());
            new_pop.push(generated_member);
        });

        self.members = new_pop;
        self.generation += 1;
        self.seed = rng.gen();
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::{
        Crossover, Fitness, FitnessRetrieve, Generate, Mutate, MutationConfig, Population,
        PopulationConfig,
    };

    impl Mutate for i64 {
        fn mutate(&self, _config: &MutationConfig, _seed: [u8; 32]) -> Self {
            4
        }
    }

    impl FitnessRetrieve for i64 {
        fn get_fitness(&self) -> Option<f64> {
            Some(5.0)
        }
    }

    impl Fitness for i64 {
        fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
            Some(6.0)
        }
    }

    impl Crossover for i64 {
        fn crossover(&self, _other: &Self, _seed: [u8; 32]) -> Self {
            2
        }
    }

    impl Generate for i64 {
        fn generate(_seed: [u8; 32]) -> Self {
            1
        }
    }

    #[test]
    fn test_serialise() {
        let config = PopulationConfig {
            pop_size: 10,
            crossover_count: 2,
            mutate_count: 2,
            elitism_count: 2,
            mutation_config: MutationConfig {
                gene_mutation_chance: 0.3,
            },
            seed: [1; 32],
            preseeded_population: vec![],
        };
        let mut p: Population<i64> = Population::new(config);
        p.tick();
        p.tick();

        let json_string = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":3,\"seed\":[194,153,183,159,34,72,149,228,138,147,183,238,122,49,198,59,71,25,17,126,81,70,27,202,105,232,186,46,14,39,39,191]}", &json_string);
    }

    impl Mutate for i32 {
        fn mutate(&self, _config: &MutationConfig, seed: [u8; 32]) -> Self {
            let mut rng: StdRng = SeedableRng::from_seed(seed);
            rng.gen()
        }
    }

    impl FitnessRetrieve for i32 {
        fn get_fitness(&self) -> Option<f64> {
            Some(5.0)
        }
    }

    impl Fitness for i32 {
        fn calculate_fitness(&mut self, seed: [u8; 32]) -> Option<f64> {
            let mut rng: StdRng = SeedableRng::from_seed(seed);
            Some(rng.gen())
        }
    }

    impl Crossover for i32 {
        fn crossover(&self, _other: &Self, seed: [u8; 32]) -> Self {
            let mut rng: StdRng = SeedableRng::from_seed(seed);
            rng.gen()
        }
    }

    impl Generate for i32 {
        fn generate(seed: [u8; 32]) -> Self {
            let mut rng: StdRng = SeedableRng::from_seed(seed);
            rng.gen()
        }
    }

    #[test]
    fn test_deterministic() {
        let config = PopulationConfig {
            pop_size: 10,
            crossover_count: 2,
            mutate_count: 2,
            elitism_count: 2,
            mutation_config: MutationConfig {
                gene_mutation_chance: 0.3,
            },
            seed: [1; 32],
            preseeded_population: vec![],
        };
        let mut p: Population<i64> = Population::new(config);

        let json_string = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,1,1,1,1,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":1,\"seed\":[61,119,195,211,231,165,151,165,122,239,25,225,34,155,137,19,36,226,231,187,28,137,64,231,241,187,37,96,44,109,235,7]}", &json_string);
        p.tick();
        let json_string_saved = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":2,\"seed\":[249,208,109,107,37,33,235,161,187,143,43,252,141,78,107,106,236,78,206,58,78,254,64,113,134,52,253,114,200,93,157,62]}", &json_string_saved);
        p.tick();
        let json_string_third = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":3,\"seed\":[194,153,183,159,34,72,149,228,138,147,183,238,122,49,198,59,71,25,17,126,81,70,27,202,105,232,186,46,14,39,39,191]}", &json_string_third);

        // Deserialise and test
        let mut p: Population<i64> = serde_json::from_str(&json_string_saved).unwrap();
        p.tick();
        let json_string_third_again = serde_json::to_string(&p).unwrap();
        assert_eq!(json_string_third, json_string_third_again);
    }

    #[test]
    fn test_deterministic_parallel_tick() {
        let config = PopulationConfig {
            pop_size: 10,
            crossover_count: 2,
            mutate_count: 2,
            elitism_count: 2,
            mutation_config: MutationConfig {
                gene_mutation_chance: 0.3,
            },
            seed: [1; 32],
            preseeded_population: vec![],
        };
        let mut p: Population<i64> = Population::new(config);

        let json_string = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,1,1,1,1,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":1,\"seed\":[61,119,195,211,231,165,151,165,122,239,25,225,34,155,137,19,36,226,231,187,28,137,64,231,241,187,37,96,44,109,235,7]}", &json_string);
        p.tick_parallel();
        let json_string_saved = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":2,\"seed\":[249,208,109,107,37,33,235,161,187,143,43,252,141,78,107,106,236,78,206,58,78,254,64,113,134,52,253,114,200,93,157,62]}", &json_string_saved);
        p.tick_parallel();
        let json_string_third = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3},\"preseeded_population\":[]},\"generation\":3,\"seed\":[194,153,183,159,34,72,149,228,138,147,183,238,122,49,198,59,71,25,17,126,81,70,27,202,105,232,186,46,14,39,39,191]}", &json_string_third);

        // Deserialise and test
        let mut p: Population<i64> = serde_json::from_str(&json_string_saved).unwrap();
        p.tick_parallel();
        let json_string_third_again = serde_json::to_string(&p).unwrap();
        assert_eq!(json_string_third, json_string_third_again);
    }
}
