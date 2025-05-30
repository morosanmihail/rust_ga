use rand::seq::SliceRandom;

// const POP_SIZE: usize = 100;
// const POP_MUT_COUNT: usize = 20;
// const POP_MUT_CHANCE: i32 = 10;
// const POP_CROSS_COUNT: i32 = 20;
// const POP_CROSS_CHANCE: i32 = 20;
// const ELITISM_COUNT: usize = 10;

#[derive(Debug, Default, Clone)]
pub struct Genome<T: Clone + Default> {
    pub data: T,
    pub fitness: Option<f64>,
}

pub trait Crossover {
    fn crossover(&self, other: &Self) -> Self;
}

pub trait Fitness {
    fn calculate_fitness(&mut self) -> Option<f64>;
}

pub trait FitnessRetrieve {
    fn get_fitness(&self) -> Option<f64>;
}

impl<T: Default + Clone> FitnessRetrieve for Genome<T> {
    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}

pub trait Mutate {
    fn mutate(&self) -> Self;
}

pub trait Generate {
    fn generate() -> Self;
}

#[derive(Debug, Default, Clone)]
pub struct PopulationConfig {
    pub pop_size: usize,
    pub crossover_count: usize,
    pub mutate_count: usize,
    pub elitism_count: usize,
}

#[derive(Debug)]
pub struct Population<T: Generate + Crossover + Mutate + Fitness + FitnessRetrieve + Default> {
    pub members: Vec<T>,
    pub config: PopulationConfig,
}

impl<T: Generate + Crossover + Mutate + Fitness + FitnessRetrieve + Default + Clone> Population<T> {
    pub fn new(config: PopulationConfig) -> Population<T> {
        let mut members: Vec<T> = Vec::new();
        for _ in 1..=config.pop_size {
            members.push(T::generate());
        }
        Population { members, config }
    }

    pub fn sort_members(&mut self) {
        self.members.sort_by(|a, b| {
            a.get_fitness()
                .partial_cmp(&b.get_fitness())
                .unwrap_or(std::cmp::Ordering::Less)
        });
    }

    pub fn get_best_member(&mut self) -> &T {
        self.sort_members();
        &self.members[0]
    }

    pub fn tick(&mut self) {
        let mut rng = rand::thread_rng();
        let mut new_pop: Vec<T> = Vec::new();

        self.members.iter_mut().for_each(|m| {
            m.calculate_fitness();
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
        println!("{}", new_pop.len());

        // Then mutation
        (0..self.config.mutate_count).for_each(|_| {
            let mutatable_member = self.members.choose(&mut rng);
            if let Some(t) = mutatable_member {
                let mut m = t.mutate();
                m.calculate_fitness();
                new_pop.push(m);
            }
        });
        println!("{}", new_pop.len());

        // Then crossover
        (0..self.config.crossover_count).for_each(|_| {
            let crossoverable_members: Vec<&T> =
                self.members.choose_multiple(&mut rng, 2).collect();
            let mut crossoverd_member =
                crossoverable_members[0].crossover(crossoverable_members[1]);
            crossoverd_member.calculate_fitness();
            new_pop.push(crossoverd_member);
        });
        println!("{}", new_pop.len());

        // Then newly generated ones
        (new_pop.len()..self.config.pop_size).for_each(|_| {
            let mut generated_member = T::generate();
            generated_member.calculate_fitness();
            new_pop.push(generated_member);
        });

        self.members = new_pop;
    }
}
