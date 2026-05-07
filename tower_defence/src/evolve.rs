use std::sync::atomic::{AtomicU64, Ordering};

use ga::{
    item_array::ItemArray,
    population::MutationConfig,
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

static MAP_SEED: AtomicU64 = AtomicU64::new(0x5EED_C0DE_CAFE_BABE);

pub fn set_map_seed(seed: u64) {
    MAP_SEED.store(seed, Ordering::Relaxed);
}

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

// ── Perlin noise ──────────────────────────────────────────────────────────────

fn make_perm(seed: u64) -> [u8; 512] {
    let mut p: [u8; 256] = core::array::from_fn(|i| i as u8);
    let mut s = seed;
    for i in (1..256).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((s >> 33) as usize) % (i + 1);
        p.swap(i, j);
    }
    let mut perm = [0u8; 512];
    for i in 0..256 { perm[i] = p[i]; perm[i + 256] = p[i]; }
    perm
}

fn perlin_fade(t: f64) -> f64 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
fn perlin_lerp(a: f64, b: f64, t: f64) -> f64 { a + t * (b - a) }
fn perlin_grad(h: u8, x: f64, y: f64) -> f64 {
    match h & 3 { 0 => x + y, 1 => -x + y, 2 => x - y, _ => -x - y }
}

fn perlin2d(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    let xi = (x.floor() as i64).rem_euclid(256) as usize;
    let yi = (y.floor() as i64).rem_euclid(256) as usize;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = perlin_fade(xf);
    let v = perlin_fade(yf);
    let a  = perm[xi]     as usize + yi;
    let b  = perm[xi + 1] as usize + yi;
    let aa = perm[a]; let ab = perm[a + 1];
    let ba = perm[b]; let bb = perm[b + 1];
    perlin_lerp(
        perlin_lerp(perlin_grad(aa, xf,       yf      ),
                    perlin_grad(ba, xf - 1.0, yf      ), u),
        perlin_lerp(perlin_grad(ab, xf,       yf - 1.0),
                    perlin_grad(bb, xf - 1.0, yf - 1.0), u),
        v,
    )
}

/// Recreates the evaluation map using the current global MAP_SEED.
/// All evaluations within one GA run share the same seed (set via set_map_seed).
pub fn make_eval_map() -> Map {
    let perm = make_perm(MAP_SEED.load(Ordering::Relaxed));
    let mut cells: Vec<Vec<Cell>> = (0..MAP_H).map(|y| {
        (0..MAP_W).map(|x| {
            let fx = x as f64 * 0.28;
            let fy = y as f64 * 0.28;
            let n1 = perlin2d(&perm, fx, fy);
            let n2 = perlin2d(&perm, fx * 2.1 + 31.7, fy * 2.1 + 17.3) * 0.45;
            let n = n1 + n2;
            let terrain = if n < -0.22 { Terrain::Water }
                          else if n > 0.28 { Terrain::Rock }
                          else { Terrain::Plain };
            Cell::new(terrain)
        }).collect()
    }).collect();

    // Force plain border so spawn points and edge pathing always work.
    for x in 0..MAP_W {
        cells[0][x].terrain = Terrain::Plain;
        cells[MAP_H - 1][x].terrain = Terrain::Plain;
    }
    for y in 0..MAP_H {
        cells[y][0].terrain = Terrain::Plain;
        cells[y][MAP_W - 1].terrain = Terrain::Plain;
    }
    // Force builder start walkable.
    cells[BUILDER_START.1][BUILDER_START.0].terrain = Terrain::Plain;

    Map::new(cells, vec![(0, 0), (MAP_W - 1, 0), (0, MAP_H - 1), (MAP_W - 1, MAP_H - 1)])
}

/// Maximum achievable fitness for the eval config.
/// Ticks survived + HP bonus + all possible enemy kills.
pub fn max_fitness() -> f64 {
    let cfg = eval_config();
    let map = make_eval_map();
    let n_spawns = map.spawn_points.len() as u64;
    let max_kills = if SIM_TICKS > cfg.spawn_delay {
        ((SIM_TICKS - cfg.spawn_delay) / cfg.spawn_interval + 1) * cfg.enemies_per_spawn as u64 * n_spawns
    } else { 0 };
    SIM_TICKS as f64 + BUILDER_HP as f64 + max_kills as f64
}

pub fn eval_config() -> Config {
    Config {
        spawn_delay: 70,
        spawn_interval: 8,
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

        let fitness = if sim.is_game_over() {
            sim.tick as f64 + sim.enemies_killed as f64
        } else {
            SIM_TICKS as f64 + sim.builder.hp as f64 + sim.enemies_killed as f64
        };

        self.0.set_fitness(Some(fitness));
        Some(fitness)
    }
}
