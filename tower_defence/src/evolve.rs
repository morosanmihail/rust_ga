use ga::{
    item_array::ItemArray,
    population::MutationConfig,
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    builder::{Builder, Instruction},
    map::{Cell, Map, Terrain},
    simulation::{Config, Simulation},
};

// Map dimensions must be const so Generate/Mutate can use them without parameters.
pub const MAP_W: usize = 20;
pub const MAP_H: usize = 12;
pub const BUILDER_START: (usize, usize) = (1, 1);
pub const BUILDER_HP: u32 = 50;

const SIM_TICKS: u64 = 300;
const MIN_INSTRS: usize = 5;
const MAX_INSTRS: usize = 50;

/// Recreates the fixed evaluation map. Always identical — fitness is deterministic.
pub fn make_eval_map() -> Map {
    let mut cells: Vec<Vec<Cell>> = (0..MAP_H)
        .map(|_| (0..MAP_W).map(|_| Cell::new(Terrain::Plain)).collect())
        .collect();
    for y in 3..=6 { cells[y][6] = Cell::new(Terrain::Rock); }
    for x in 9..=12 { cells[5][x] = Cell::new(Terrain::Water); }
    Map::new(cells, vec![(0, 0), (19, 0), (0, 11), (19, 11)])
}

pub fn eval_config() -> Config {
    Config {
        spawn_delay: 70,
        spawn_interval: 15,
        enemies_per_spawn: 1,
        enemy_hp: 32,
        enemy_damage: 2,
        tower_range: 4.0,
        tower_damage: 4,
        tower_hp: 20,
        wall_hp: 15,
        builder_hp: BUILDER_HP,
    }
}

// ── Gene ─────────────────────────────────────────────────────────────────────

/// One gene = one builder instruction encoded as (kind, x, y).
/// kind: 0=Move  1=BuildTower  2=BuildWall  3=BuildBridge
#[derive(Clone, Copy, Debug, Default)]
pub struct GeneInstruction {
    pub kind: u8,
    pub x: u8,
    pub y: u8,
}

impl GeneInstruction {
    pub fn to_instruction(&self) -> Instruction {
        let x = (self.x as usize).min(MAP_W - 1);
        let y = (self.y as usize).min(MAP_H - 1);
        match self.kind % 4 {
            0 => Instruction::Move(x, y),
            1 => Instruction::BuildTower(x, y),
            2 => Instruction::BuildWall(x, y),
            _ => Instruction::BuildBridge(x, y),
        }
    }
}

impl Generate for GeneInstruction {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        GeneInstruction {
            kind: rng.gen_range(0..4u8),
            x:    rng.gen_range(0..MAP_W as u8),
            y:    rng.gen_range(0..MAP_H as u8),
        }
    }
}

impl Mutate for GeneInstruction {
    fn mutate(&self, _config: &MutationConfig, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        match rng.gen_range(0..3u8) {
            // change instruction type only
            0 => GeneInstruction { kind: rng.gen_range(0..4u8), ..*self },
            // nudge coordinates
            1 => GeneInstruction {
                x: (self.x as i32 + rng.gen_range(-3..=3i32)).clamp(0, MAP_W as i32 - 1) as u8,
                y: (self.y as i32 + rng.gen_range(-3..=3i32)).clamp(0, MAP_H as i32 - 1) as u8,
                ..*self
            },
            // completely new random instruction
            _ => GeneInstruction::generate(rng.gen()),
        }
    }
}

// ── Genome ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct BuilderGenome(pub ItemArray<GeneInstruction>);

impl BuilderGenome {
    pub fn to_instructions(&self) -> Vec<Instruction> {
        self.0.get_data().iter().map(|g| g.to_instruction()).collect()
    }

    pub fn len(&self) -> usize {
        self.0.get_data().len()
    }
}

impl Generate for BuilderGenome {
    fn generate(seed: [u8; 32]) -> Self {
        BuilderGenome(ItemArray::generate_length(MIN_INSTRS, MAX_INSTRS, seed))
    }
}

impl Mutate for BuilderGenome {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        BuilderGenome(self.0.mutate(config, seed))
    }
}

impl Crossover for BuilderGenome {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        BuilderGenome(self.0.crossover(&other.0, seed))
    }
}

impl FitnessRetrieve for BuilderGenome {
    fn get_fitness(&self) -> Option<f64> {
        self.0.get_fitness()
    }
}

impl Fitness for BuilderGenome {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        if let Some(f) = self.0.get_fitness() {
            return Some(f);
        }

        let config = eval_config();
        let mut builder = Builder::new(BUILDER_START.0, BUILDER_START.1, BUILDER_HP);
        for instr in self.to_instructions() {
            builder.push(instr);
        }
        let mut sim = Simulation::new(make_eval_map(), builder, config);

        for _ in 0..SIM_TICKS {
            sim.tick();
            if sim.is_game_over() { break; }
        }

        // Fitness: ticks survived. Surviving all 300 ticks earns an HP bonus.
        let fitness = if sim.is_game_over() {
            sim.tick as f64
        } else {
            SIM_TICKS as f64 + sim.builder.hp as f64
        };

        self.0.set_fitness(Some(fitness));
        Some(fitness)
    }
}
