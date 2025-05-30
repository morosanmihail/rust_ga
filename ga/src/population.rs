use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate};

#[derive(Debug, Default, Clone)]
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
pub struct PopulationConfig {
    pub seed: [u8; 32],
    pub pop_size: usize,
    pub crossover_count: usize,
    pub mutate_count: usize,
    pub elitism_count: usize,

    pub mutation_config: MutationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Population<T: Generate + Crossover + Mutate + Fitness + FitnessRetrieve + Default> {
    pub members: Vec<T>,
    pub config: PopulationConfig,
    generation: i64,
    seed: [u8; 32],
}

impl<T: Generate + Crossover + Mutate + Fitness + FitnessRetrieve + Default + Clone> Population<T> {
    pub fn new(config: PopulationConfig) -> Population<T> {
        let mut rng: StdRng = SeedableRng::from_seed(config.seed);
        let mut members: Vec<T> = Vec::new();
        for _ in 1..=config.pop_size {
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

    pub fn tick(&mut self) {
        let mut rng: StdRng = SeedableRng::from_seed(self.seed);
        let mut new_pop: Vec<T> = Vec::new();

        self.members.iter_mut().for_each(|m| {
            m.calculate_fitness(rng.gen());
        });
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
        };
        let mut p: Population<i64> = Population::new(config);
        p.tick();
        p.tick();

        let json_string = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3}},\"generation\":3}", &json_string);
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
        };
        let mut p: Population<i64> = Population::new(config);

        let json_string = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,1,1,1,1,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3}},\"generation\":1,\"seed\":[61,119,195,211,231,165,151,165,122,239,25,225,34,155,137,19,36,226,231,187,28,137,64,231,241,187,37,96,44,109,235,7]}", &json_string);
        p.tick();
        let json_string_saved = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3}},\"generation\":2,\"seed\":[62,237,20,223,252,169,243,175,40,214,53,17,190,190,202,51,248,78,220,247,106,111,146,223,129,95,220,120,28,166,42,182]}", &json_string_saved);
        p.tick();
        let json_string_third = serde_json::to_string(&p).unwrap();
        assert_eq!("{\"members\":[1,1,4,4,2,2,1,1,1,1],\"config\":{\"seed\":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\"pop_size\":10,\"crossover_count\":2,\"mutate_count\":2,\"elitism_count\":2,\"mutation_config\":{\"gene_mutation_chance\":0.3}},\"generation\":3,\"seed\":[61,240,161,170,168,40,224,71,3,3,129,86,151,76,130,42,28,222,7,123,91,195,241,6,231,203,202,179,218,241,247,167]}", &json_string_third);

        // Deserialise and test
        let mut p: Population<i64> = serde_json::from_str(&json_string_saved).unwrap();
        p.tick();
        let json_string_third_again = serde_json::to_string(&p).unwrap();
        assert_eq!(json_string_third, json_string_third_again);
    }
}
