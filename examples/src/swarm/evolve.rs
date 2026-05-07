use std::sync::{OnceLock, RwLock};

use ga::{
    population::MutationConfig,
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::sim::{SteeringWeights, SwarmSim, WORLD_W};

// How many ticks to simulate forward when scoring a candidate genome.
const LOOKAHEAD: usize = 30;

static SIM_SNAPSHOT: OnceLock<RwLock<SwarmSim>> = OnceLock::new();

pub fn update_snapshot(sim: &SwarmSim) {
    let lock = SIM_SNAPSHOT.get_or_init(|| RwLock::new(sim.clone()));
    *lock.write().unwrap() = sim.clone();
}

fn get_snapshot() -> SwarmSim {
    SIM_SNAPSHOT.get().unwrap().read().unwrap().clone()
}

/// Genome: four steering weights [goal_pull, obstacle_push, neighbor_align, neighbor_separate].
#[derive(Clone, Debug)]
pub struct SteeringGenome {
    pub weights: [f32; 4],
    fitness: Option<f64>,
}

impl Default for SteeringGenome {
    fn default() -> Self {
        SteeringGenome { weights: [1.0, 1.0, 0.5, 0.5], fitness: None }
    }
}

impl SteeringGenome {
    pub fn to_weights(&self) -> SteeringWeights {
        SteeringWeights {
            goal_pull: self.weights[0] as f64,
            obstacle_push: self.weights[1] as f64,
            neighbor_align: self.weights[2] as f64,
            neighbor_separate: self.weights[3] as f64,
        }
    }
}

impl Generate for SteeringGenome {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        SteeringGenome {
            weights: core::array::from_fn(|_| rng.gen_range(0.0f32..3.0)),
            fitness: None,
        }
    }
}

impl Mutate for SteeringGenome {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let weights = core::array::from_fn(|i| {
            if rng.gen::<f64>() < config.gene_mutation_chance {
                (self.weights[i] + rng.gen_range(-0.5f32..0.5)).clamp(0.0, 5.0)
            } else {
                self.weights[i]
            }
        });
        SteeringGenome { weights, fitness: None }
    }
}

impl Crossover for SteeringGenome {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let point = rng.gen_range(0..=4usize);
        let weights = core::array::from_fn(|i| {
            if i < point { self.weights[i] } else { other.weights[i] }
        });
        SteeringGenome { weights, fitness: None }
    }
}

impl FitnessRetrieve for SteeringGenome {
    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}

impl Fitness for SteeringGenome {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        // Always re-evaluate from the current snapshot — snapshot changes every sim tick.
        let mut sim = get_snapshot();
        let w = self.to_weights();
        for _ in 0..LOOKAHEAD {
            sim.step(&w);
            if sim.agents_at_goal() == sim.agents.len() {
                break;
            }
        }
        // Per-agent progress sum, large goal bonus, and a heavy worst-agent term.
        // Worst-agent term dominates once most agents are done, so the GA cannot
        // ignore the last straggler.
        let min_progress = sim
            .agents
            .iter()
            .map(|a| if a.reached_goal { WORLD_W } else { a.pos.0 })
            .fold(f64::INFINITY, f64::min);
        let score = sim.progress_score() / WORLD_W * 60.0
            + sim.agents_at_goal() as f64 * 50.0
            + min_progress / WORLD_W * 80.0;
        self.fitness = Some(score);
        Some(score)
    }
}
