use ga::population::{MutationConfig, Population, PopulationConfig};
use ga::traits::FitnessRetrieve;
use rand::Rng;

use tower_defence::{
    builder::Builder,
    evolve::{eval_config, make_eval_map, max_fitness, BuilderGenome, BUILDER_START, BUILDER_HP},
    render::render,
    simulation::Simulation,
};

const GENERATIONS: usize = 200;
const POP_SIZE: usize = 80;

fn main() {
    println!("=== Tower Defence — Evolving Builder Instructions ===\n");
    println!("Map: 20×12, 4 corner spawns, Perlin-noise terrain (deterministic seed)");
    println!("Fitness: ticks survived (max 300) + remaining HP if all 300 survived");
    println!("Population: {} | Generations: {}\n", POP_SIZE, GENERATIONS);

    let config = PopulationConfig {
        pop_size: POP_SIZE,
        crossover_count: 25,
        mutate_count: 25,
        elitism_count: 5,
        mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
        seed: rand::thread_rng().gen(),
        preseeded_population: vec![],
    };

    let mut pop: Population<BuilderGenome> = Population::new(config);

    let MAX_FITNESS: f64 = max_fitness();
    let mut solved_gen = None;

    for gen in 1..=GENERATIONS {
        pop.tick_parallel();
        let best = pop.get_best_member();
        let fitness = best.get_fitness().unwrap_or(0.0);
        let len = best.len();
        if gen % 10 == 0 || gen <= 5 || fitness >= MAX_FITNESS {
            println!("Gen {:>4}: best fitness = {:>7.1}  instrs = {}", gen, fitness, len);
        }
        if fitness >= MAX_FITNESS {
            solved_gen = Some(gen);
            break;
        }
    }

    match solved_gen {
        Some(g) => println!("\n=== Max fitness reached at generation {} ===\n", g),
        None    => println!("\n=== Evolution complete (max fitness not reached) ===\n"),
    }

    // Extract best genome and replay with rendering.
    let best = pop.get_best_member().clone();
    let instrs = best.to_instructions();
    println!("Best genome: {} instructions, fitness = {:.1}", instrs.len(), best.get_fitness().unwrap_or(0.0));
    println!("Instructions:");
    for (i, instr) in instrs.iter().enumerate() {
        println!("  {:>2}. {:?}", i + 1, instr);
    }
    println!();

    // Replay the best solution with full rendering.
    let mut builder = Builder::new(BUILDER_START.0, BUILDER_START.1, BUILDER_HP);
    for instr in instrs {
        builder.push(instr);
    }
    let mut sim = Simulation::new(make_eval_map(), builder, eval_config());

    render(&sim);
    for _ in 0..300 {
        sim.tick();
        let in_combat = sim.tick >= sim.config.spawn_delay || !sim.enemies.is_empty();
        if in_combat || sim.tick % 10 == 0 {
            render(&sim);
        }
        if sim.is_game_over() {
            println!("=== GAME OVER at tick {} ===", sim.tick);
            break;
        }
    }

    if !sim.is_game_over() {
        println!("=== SURVIVED all 300 ticks! Builder HP: {} ===", sim.builder.hp);
    }
}
