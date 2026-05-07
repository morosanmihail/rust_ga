mod swarm;

use std::thread::sleep;
use std::time::Duration;

use ga::population::{MutationConfig, Population, PopulationConfig};

use swarm::evolve::{update_snapshot, SteeringGenome};
use swarm::sim::{SwarmSim, WORLD_H, WORLD_W};

const GA_GENS_PER_TICK: usize = 20;
const MAX_SIM_TICKS: u64 = 600;
const FRAME_MS: u64 = 60;

fn render(sim: &SwarmSim, best: &SteeringGenome, total_ga_gens: u64) {
    let gw = WORLD_W as usize;
    let gh = WORLD_H as usize;
    let mut grid = vec![vec!['.'; gw]; gh];

    for obs in &sim.obstacles {
        for gy in 0..gh {
            for gx in 0..gw {
                let dx = gx as f64 - obs.pos.0;
                let dy = gy as f64 - obs.pos.1;
                if dx * dx + dy * dy <= obs.radius * obs.radius {
                    grid[gy][gx] = '#';
                }
            }
        }
    }

    let g = &sim.goal;
    let gx0 = (g.x as usize).clamp(0, gw - 1);
    let gx1 = ((g.x + g.w) as usize).clamp(0, gw - 1);
    let gy0 = (g.y as usize).clamp(0, gh - 1);
    let gy1 = ((g.y + g.h) as usize).clamp(0, gh - 1);
    for gy in gy0..=gy1 {
        for gx in gx0..=gx1 {
            if grid[gy][gx] == '.' { grid[gy][gx] = 'G'; }
        }
    }

    for agent in &sim.agents {
        if agent.reached_goal {
            continue;
        }
        let gx = (agent.pos.0.round() as usize).clamp(0, gw - 1);
        let gy = (agent.pos.1.round() as usize).clamp(0, gh - 1);
        grid[gy][gx] = '@';
    }

    // Stack finished agents along the right edge.
    let n_done = sim.agents_at_goal();
    for i in 0..n_done.min(gh) {
        grid[i][gw - 1] = '*';
    }

    print!("\x1b[2J\x1b[H");
    println!("=== SWARM STEERING  (per-tick GA) ===");
    println!(
        "Sim tick {:>4}  |  Goal: {:>2}/{:<2}  |  GA gens so far: {}",
        sim.tick,
        n_done,
        sim.agents.len(),
        total_ga_gens
    );
    let bw = best.weights;
    println!(
        "Best weights  goal={:.2}  obs={:.2}  align={:.2}  sep={:.2}",
        bw[0], bw[1], bw[2], bw[3]
    );
    println!("Legend:  @ agent   # obstacle   G goal   * done");
    println!();
    for row in &grid {
        let s: String = row.iter().collect();
        println!("{}", s);
    }
    println!();
}

fn main() {
    let mut sim = SwarmSim::new();
    update_snapshot(&sim);

    let config: PopulationConfig<SteeringGenome> = PopulationConfig {
        seed: [13u8; 32],
        pop_size: 40,
        elitism_count: 4,
        mutate_count: 16,
        crossover_count: 10,
        mutation_config: MutationConfig { gene_mutation_chance: 0.5 },
        preseeded_population: vec![],
    };

    let mut pop: Population<SteeringGenome> = Population::new(config);
    let mut total_ga_gens: u64 = 0;

    loop {
        for _ in 0..GA_GENS_PER_TICK {
            pop.tick_parallel();
            total_ga_gens += 1;
        }

        let best = pop.get_best_member().clone();
        sim.step(&best.to_weights());
        update_snapshot(&sim);

        render(&sim, &best, total_ga_gens);

        if sim.agents_at_goal() == sim.agents.len() {
            println!(
                "All {} agents reached goal at tick {}!",
                sim.agents.len(),
                sim.tick
            );
            break;
        }
        if sim.tick >= MAX_SIM_TICKS {
            println!(
                "Timeout at tick {}. {}/{} agents reached goal.",
                sim.tick,
                sim.agents_at_goal(),
                sim.agents.len()
            );
            break;
        }

        sleep(Duration::from_millis(FRAME_MS));
    }
}
