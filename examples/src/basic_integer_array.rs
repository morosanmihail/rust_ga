use ga::{
    item_array::ItemArray,
    population::{MutationConfig, Population, PopulationConfig},
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Copy, Default, Debug)]
struct Integer(i64);

#[derive(Clone, Debug)]
struct IntegerArray(ItemArray<Integer>);

const MIN_VALUE: i64 = -255;
const MAX_VALUE: i64 = 255;

impl Fitness for IntegerArray {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        if self.0.get_fitness().is_none() {
            let res: i64 = self
                .0
                .get_data()
                .iter()
                .map(|v| if v.0 == 0 { 1 } else { 0 })
                .sum();
            self.0.set_fitness(Some(res as f64));
            return Some(res as f64);
        }
        self.0.get_fitness()
    }
}

impl Generate for Integer {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        Integer(rng.gen_range(MIN_VALUE..=MAX_VALUE))
    }
}

impl Mutate for Integer {
    fn mutate(&self, _config: &MutationConfig, _seed: [u8; 32]) -> Self {
        let mut rng = rand::thread_rng();
        Integer(self.0 + rng.gen_range(MIN_VALUE / 10..=MAX_VALUE / 10))
    }
}

impl Generate for IntegerArray {
    fn generate(seed: [u8; 32]) -> Self {
        IntegerArray(ItemArray::generate_length(2, 4, seed))
        // IntegerArray(ItemArray::generate())
    }
}

impl Crossover for IntegerArray {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        IntegerArray(self.0.crossover(&other.0, seed))
    }
}

impl FitnessRetrieve for IntegerArray {
    fn get_fitness(&self) -> Option<f64> {
        self.0.get_fitness()
    }
}

impl Mutate for IntegerArray {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        IntegerArray(self.0.mutate(config, seed))
    }
}

impl Default for IntegerArray {
    fn default() -> Self {
        IntegerArray(ItemArray::default())
    }
}

fn main() {
    let config = PopulationConfig {
        pop_size: 10,
        crossover_count: 2,
        mutate_count: 2,
        elitism_count: 2,
        mutation_config: MutationConfig {
            gene_mutation_chance: 0.3,
        },
        seed: rand::thread_rng().gen(),
    };
    let mut p: Population<IntegerArray> = Population::new(config);

    (0..10000).for_each(|i| {
        p.tick();
        let best = p.get_best_member();
        println!("Gen {i}: {:?}", best);
    });
}
