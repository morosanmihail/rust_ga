#[allow(dead_code)]
mod tower_defence;

/// Calibration tool: run with ENEMY_HP=<n> to measure GA convergence speed.
/// Prints gen-by-gen best fitness and average gens to reach max fitness (350).
///
/// Target: avg 15-20 gens. Empirical result: HP=32 → avg ~29 gens, 9/10 runs solve in 50 gens.
use ga::item_array::ItemArray;
use ga::population::{MutationConfig, Population, PopulationConfig};
use ga::traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate};
use rand::Rng;

use tower_defence::builder::Builder;
use tower_defence::evolve::{GeneInstruction, make_eval_map, BUILDER_HP, BUILDER_START};
use tower_defence::simulation::{Config, Simulation};

const TEST_GENERATIONS: usize = 50;
const POP_SIZE: usize = 80;
const RUNS: usize = 10;
const SIM_TICKS: u64 = 300;

fn enemy_hp() -> u32 {
    std::env::var("ENEMY_HP").ok().and_then(|s| s.parse().ok()).unwrap_or(32)
}

fn make_config(hp: u32) -> Config {
    Config {
        spawn_delay: 70,
        spawn_interval: 15,
        enemies_per_spawn: 1,
        enemy_hp: hp,
        enemy_damage: 2,
        tower_range: 4.0,
        tower_damage: 4,
        tower_hp: 20,
        wall_hp: 15,
    }
}

#[derive(Clone, Debug, Default)]
struct TestGenome {
    inner: ItemArray<GeneInstruction>,
    enemy_hp: u32,
}

impl Generate for TestGenome {
    fn generate(seed: [u8; 32]) -> Self {
        TestGenome { inner: ItemArray::generate_length(5, 50, seed), enemy_hp: enemy_hp() }
    }
}
impl Mutate for TestGenome {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        TestGenome { inner: self.inner.mutate(config, seed), enemy_hp: self.enemy_hp }
    }
}
impl Crossover for TestGenome {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        TestGenome { inner: self.inner.crossover(&other.inner, seed), enemy_hp: self.enemy_hp }
    }
}
impl FitnessRetrieve for TestGenome {
    fn get_fitness(&self) -> Option<f64> { self.inner.get_fitness() }
}
impl Fitness for TestGenome {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        if let Some(f) = self.inner.get_fitness() { return Some(f); }
        let mut builder = Builder::new(BUILDER_START.0, BUILDER_START.1, BUILDER_HP);
        for instr in self.inner.get_data().iter().map(|g| g.to_instruction()) {
            builder.push(instr);
        }
        let mut sim = Simulation::new(make_eval_map(), builder, make_config(self.enemy_hp));
        for _ in 0..SIM_TICKS { sim.tick(); if sim.is_game_over() { break; } }
        let fitness = if sim.is_game_over() { sim.tick as f64 + sim.enemies_killed as f64 } else { SIM_TICKS as f64 + sim.builder.hp as f64 + sim.enemies_killed as f64 };
        self.inner.set_fitness(Some(fitness));
        Some(fitness)
    }
}

fn main() {
    let hp = enemy_hp();
    let max_fitness = SIM_TICKS as f64 + BUILDER_HP as f64;
    println!("ENEMY_HP={hp}  max_fitness={max_fitness}  pop={POP_SIZE}  gens={TEST_GENERATIONS}  runs={RUNS}");

    let mut rng = rand::thread_rng();
    let mut solved_gens: Vec<Option<usize>> = Vec::new();

    for run in 0..RUNS {
        let seed: [u8; 32] = rng.gen();
        let config = PopulationConfig {
            pop_size: POP_SIZE,
            crossover_count: 25,
            mutate_count: 25,
            elitism_count: 5,
            mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
            seed,
            preseeded_population: vec![],
        };
        let mut pop: Population<TestGenome> = Population::new(config);
        let history: Vec<f64> = (0..TEST_GENERATIONS).map(|_| {
            pop.tick_parallel();
            pop.get_best_member().get_fitness().unwrap_or(0.0)
        }).collect();

        let solved = history.iter().position(|&f| f >= max_fitness).map(|g| g + 1);
        let summary: Vec<String> = history.iter().enumerate()
            .filter(|(g, _)| *g == 0 || history[*g] > history[g - 1] || *g == TEST_GENERATIONS - 1)
            .map(|(g, f)| format!("g{}={:.0}", g + 1, f))
            .collect();
        println!("  run {:>2}: [{}]  solved={:?}", run + 1, summary.join(" "), solved);
        solved_gens.push(solved);
    }

    let solved: Vec<usize> = solved_gens.iter().filter_map(|x| *x).collect();
    let avg = if solved.is_empty() { f64::INFINITY }
              else { solved.iter().sum::<usize>() as f64 / solved.len() as f64 };
    println!("  solved {}/{} runs, avg gen to solve = {:.1}", solved.len(), RUNS, avg);
}
