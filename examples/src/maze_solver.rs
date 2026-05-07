use std::{collections::HashSet, sync::OnceLock};

use ga::{
    item_array::ItemArray,
    population::{MutationConfig, Population, PopulationConfig},
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const MAZE_W: usize = 24;
const MAZE_H: usize = 24;
const PRIZE_COUNT: usize = 10;
const PRIZE_VALUE: f64 = 50.0;
const SOLVED_BONUS: f64 = 100.0;
const STEP_PENALTY: f64 = 0.1;
const EXTRA_DECISION_PENALTY: f64 = 1.0;

// ── Directions ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    N = 0,
    E = 1,
    S = 2,
    W = 3,
}

impl Dir {
    const ALL: [Dir; 4] = [Dir::N, Dir::E, Dir::S, Dir::W];

    fn opposite(self) -> Dir {
        match self {
            Dir::N => Dir::S,
            Dir::E => Dir::W,
            Dir::S => Dir::N,
            Dir::W => Dir::E,
        }
    }

    fn left(self) -> Dir {
        match self {
            Dir::N => Dir::W,
            Dir::E => Dir::N,
            Dir::S => Dir::E,
            Dir::W => Dir::S,
        }
    }

    fn right(self) -> Dir {
        match self {
            Dir::N => Dir::E,
            Dir::E => Dir::S,
            Dir::S => Dir::W,
            Dir::W => Dir::N,
        }
    }

    fn delta(self) -> (isize, isize) {
        match self {
            Dir::N => (0, -1),
            Dir::E => (1, 0),
            Dir::S => (0, 1),
            Dir::W => (-1, 0),
        }
    }
}

// ── Maze ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct MazeData {
    width: usize,
    height: usize,
    walls: Vec<[bool; 4]>, // per cell: [N, E, S, W], true = wall present
    prizes: Vec<(usize, usize)>,
}

impl MazeData {
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn wall(&self, x: usize, y: usize, d: Dir) -> bool {
        self.walls[self.idx(x, y)][d as usize]
    }

    fn end(&self) -> (usize, usize) {
        (self.width - 1, self.height - 1)
    }
}

fn build_maze(width: usize, height: usize, rng: &mut StdRng) -> MazeData {
    let mut walls = vec![[true; 4]; width * height];
    let mut visited = vec![false; width * height];
    let idx = |x: usize, y: usize| y * width + x;

    let mut stack = vec![(0usize, 0usize)];
    visited[0] = true;

    while let Some(&(cx, cy)) = stack.last() {
        let candidates: Vec<(usize, usize, Dir)> = Dir::ALL
            .iter()
            .filter_map(|&d| {
                let (dx, dy) = d.delta();
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && ny >= 0 && nx < width as isize && ny < height as isize {
                    let (nx, ny) = (nx as usize, ny as usize);
                    if !visited[idx(nx, ny)] {
                        return Some((nx, ny, d));
                    }
                }
                None
            })
            .collect();

        if candidates.is_empty() {
            stack.pop();
        } else {
            let (nx, ny, d) = candidates[rng.gen_range(0..candidates.len())];
            walls[idx(cx, cy)][d as usize] = false;
            walls[idx(nx, ny)][d.opposite() as usize] = false;
            visited[idx(nx, ny)] = true;
            stack.push((nx, ny));
        }
    }

    let (end_x, end_y) = (width - 1, height - 1);
    let mut prizes = Vec::new();
    let mut tries = 0;
    while prizes.len() < PRIZE_COUNT && tries < 100_000 {
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        if (x, y) != (0, 0) && (x, y) != (end_x, end_y) && !prizes.contains(&(x, y)) {
            prizes.push((x, y));
        }
        tries += 1;
    }

    MazeData { width, height, walls, prizes }
}

static MAZE: OnceLock<MazeData> = OnceLock::new();

// ── Path simulation ──────────────────────────────────────────────────────────

// Returns (score, path_cells, solved).
fn simulate(maze: &MazeData, decisions: &[Decision]) -> (f64, Vec<(usize, usize)>, bool) {
    let (end_x, end_y) = maze.end();
    let max_dist = (maze.width + maze.height - 2) as f64;

    let mut pos = (0usize, 0usize);
    let mut facing = Dir::E;
    let mut back_dir: Option<Dir> = None;
    let mut dec_i = 0usize;
    let mut prizes_collected = 0usize;
    let mut collected = vec![false; maze.prizes.len()];
    let mut path = vec![pos];

    loop {
        let (x, y) = pos;

        if x == end_x && y == end_y {
            let steps = path.len() as f64;
            let unused = (decisions.len() - dec_i) as f64;
            let score = max_dist + SOLVED_BONUS + prizes_collected as f64 * PRIZE_VALUE
                - steps * STEP_PENALTY
                - unused * EXTRA_DECISION_PENALTY;
            return (score, path, true);
        }

        for (i, &(px, py)) in maze.prizes.iter().enumerate() {
            if px == x && py == y && !collected[i] {
                collected[i] = true;
                prizes_collected += 1;
            }
        }

        // Open exits excluding where we came from.
        let choices: Vec<Dir> = Dir::ALL
            .iter()
            .copied()
            .filter(|&d| Some(d) != back_dir && !maze.wall(x, y, d))
            .collect();

        if choices.is_empty() {
            // Dead end — turn around without consuming a gene.
            if let Some(back) = back_dir {
                let (dx, dy) = back.delta();
                pos = ((x as isize + dx) as usize, (y as isize + dy) as usize);
                back_dir = Some(back.opposite());
                path.push(pos);
            }
            continue;
        }

        let chosen = if choices.len() == 1 {
            choices[0]
        } else {
            // Junction: consume a decision gene.
            if dec_i >= decisions.len() {
                break;
            }
            let gene = decisions[dec_i].0; // 0, 1, or 2
            dec_i += 1;

            // Gene 3 = turn around (go back the way we came).
            if gene == 3 {
                back_dir.unwrap_or(choices[0])
            } else {
                // Sort available choices as [left, forward, right] relative to current facing.
                let order = [facing.left(), facing, facing.right()];
                let sorted: Vec<Dir> =
                    order.iter().copied().filter(|d| choices.contains(d)).collect();

                match sorted.len() {
                    0 => choices[0],
                    1 => sorted[0],
                    // 2 choices (3-way junction): 0 = left, 1 or 2 = right
                    2 => {
                        if gene == 0 {
                            sorted[0]
                        } else {
                            sorted[1]
                        }
                    }
                    // 3 choices (4-way junction): 0 = left, 1 = forward, 2 = right
                    _ => sorted[gene as usize],
                }
            }
        };

        facing = chosen;
        let (dx, dy) = chosen.delta();
        pos = ((x as isize + dx) as usize, (y as isize + dy) as usize);
        back_dir = Some(chosen.opposite());
        path.push(pos);

        if path.len() > maze.width * maze.height + 10 {
            break; // Safety — should never trigger in a perfect maze.
        }
    }

    let (x, y) = pos;
    let manhattan =
        ((x as isize - end_x as isize).abs() + (y as isize - end_y as isize).abs()) as f64;
    let steps = path.len() as f64;
    let unused = (decisions.len() - dec_i) as f64;
    let score = (max_dist - manhattan) + prizes_collected as f64 * PRIZE_VALUE
        - steps * STEP_PENALTY
        - unused * EXTRA_DECISION_PENALTY;
    (score, path, false)
}

// ── Gene ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Decision(u8);

impl Generate for Decision {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        Decision(rng.gen_range(0u8..4))
    }
}

impl Mutate for Decision {
    fn mutate(&self, _c: &MutationConfig, seed: [u8; 32]) -> Self {
        Decision::generate(seed)
    }
}

// ── Genome ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct MazePath(ItemArray<Decision>);

impl Generate for MazePath {
    fn generate(seed: [u8; 32]) -> Self {
        MazePath(ItemArray::generate_length(10, 300, seed))
    }
}

impl Mutate for MazePath {
    fn mutate(&self, c: &MutationConfig, seed: [u8; 32]) -> Self {
        MazePath(self.0.mutate(c, seed))
    }
}

impl Crossover for MazePath {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        MazePath(self.0.crossover(&other.0, seed))
    }
}

impl FitnessRetrieve for MazePath {
    fn get_fitness(&self) -> Option<f64> {
        self.0.get_fitness()
    }
}

impl Fitness for MazePath {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        if self.0.get_fitness().is_some() {
            return self.0.get_fitness();
        }
        let maze = MAZE.get().expect("MAZE must be initialised before fitness evaluation");
        let (score, _, _) = simulate(maze, self.0.get_data());
        self.0.set_fitness(Some(score));
        Some(score)
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

fn print_maze(maze: &MazeData, path: &[(usize, usize)]) {
    let on_path: HashSet<(usize, usize)> = path.iter().copied().collect();
    let prize_pos: HashSet<(usize, usize)> = maze.prizes.iter().copied().collect();
    let (ex, ey) = maze.end();

    for row in 0..maze.height {
        // Top wall row
        print!("+");
        for col in 0..maze.width {
            print!("{}", if maze.wall(col, row, Dir::N) { "---" } else { "   " });
            print!("+");
        }
        println!();

        // Cell content row
        for col in 0..maze.width {
            print!("{}", if maze.wall(col, row, Dir::W) { "|" } else { " " });
            let ch = if col == 0 && row == 0 {
                'S'
            } else if col == ex && row == ey {
                'E'
            } else if prize_pos.contains(&(col, row)) && on_path.contains(&(col, row)) {
                '*' // prize collected by path
            } else if prize_pos.contains(&(col, row)) {
                '$' // prize not visited
            } else if on_path.contains(&(col, row)) {
                '.'
            } else {
                ' '
            };
            print!(" {} ", ch);
        }
        println!("|");
    }

    // Bottom wall
    print!("+");
    for _ in 0..maze.width {
        print!("---+");
    }
    println!();
    println!("  S=start  E=exit  $=prize  *=prize(on path)  .=path");
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let master_seed: [u8; 32] = rand::thread_rng().gen();
    let mut rng: StdRng = SeedableRng::from_seed(master_seed);

    println!("Building {}×{} maze with {} prizes…", MAZE_W, MAZE_H, PRIZE_COUNT);
    let maze = build_maze(MAZE_W, MAZE_H, &mut rng);
    let prize_count = maze.prizes.len();
    println!("Placed {prize_count} prizes. Exit at ({},{}).", MAZE_W - 1, MAZE_H - 1);
    println!(
        "Fitness: proximity_score (0–{}) + prizes×{PRIZE_VALUE} + solved_bonus({SOLVED_BONUS})\n",
        MAZE_W + MAZE_H - 2
    );
    print_maze(&maze, &[]);

    MAZE.set(maze).expect("Failed to set global maze");

    let ga_config = PopulationConfig {
        pop_size: 300,
        crossover_count: 80,
        mutate_count: 80,
        elitism_count: 20,
        mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
        seed: rng.gen(),
        preseeded_population: vec![],
    };

    let mut population: Population<MazePath> = Population::new(ga_config);

    const TOTAL_GENS: usize = 3000;
    let mut first_solve_gen: Option<usize> = None;

    println!("Evolving for {TOTAL_GENS} generations…\n");

    for gen in 0..TOTAL_GENS {
        population.tick();

        let is_last = gen == TOTAL_GENS - 1;

        if gen % 25 == 0 || is_last {
            let best = population.get_best_member();
            let maze = MAZE.get().unwrap();
            let (score, path, solved) = simulate(maze, best.0.get_data());
            let decisions: Vec<u8> = best.0.get_data().iter().map(|d| d.0).collect();
            println!(
                "Gen {:4}  score={:6.2}  steps={:3}  solved={}",
                gen + 1,
                score,
                path.len(),
                solved
            );
            println!("  decisions[{}]: {:?}", decisions.len(), decisions);
            print_maze(maze, &path);

            if solved && first_solve_gen.is_none() {
                first_solve_gen = Some(gen + 1);
                println!("── First solution at generation {} ──", gen + 1);
            }
        }
    }

    match first_solve_gen {
        Some(g) => println!("\nMaze first solved at generation {g}."),
        None => println!("\nMaze was not solved in {TOTAL_GENS} generations."),
    }
}
