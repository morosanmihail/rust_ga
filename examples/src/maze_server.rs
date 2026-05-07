use std::sync::{Arc, OnceLock, RwLock};

use axum::{extract::State, response::Html, routing::get, Json, Router};
use ga::{
    item_array::ItemArray,
    population::{MutationConfig, Population, PopulationConfig},
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

const MAZE_W: usize = 24;
const MAZE_H: usize = 24;
const PRIZE_COUNT: usize = 10;
const PRIZE_VALUE: f64 = 50.0;
const SOLVED_BONUS: f64 = 100.0;
const STEP_PENALTY: f64 = 0.1;
const EXTRA_DECISION_PENALTY: f64 = 1.0;
const GENS_PER_MAZE: usize = 100;

// ── Directions ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir { N = 0, E = 1, S = 2, W = 3 }

impl Dir {
    const ALL: [Dir; 4] = [Dir::N, Dir::E, Dir::S, Dir::W];
    fn opposite(self) -> Dir { match self { Dir::N=>Dir::S, Dir::E=>Dir::W, Dir::S=>Dir::N, Dir::W=>Dir::E } }
    fn left(self) -> Dir    { match self { Dir::N=>Dir::W, Dir::E=>Dir::N, Dir::S=>Dir::E, Dir::W=>Dir::S } }
    fn right(self) -> Dir   { match self { Dir::N=>Dir::E, Dir::E=>Dir::S, Dir::S=>Dir::W, Dir::W=>Dir::N } }
    fn delta(self) -> (isize, isize) { match self { Dir::N=>(0,-1), Dir::E=>(1,0), Dir::S=>(0,1), Dir::W=>(-1,0) } }
}

// ── Maze ─────────────────────────────────────────────────────────────────────

struct MazeData {
    width: usize,
    height: usize,
    walls: Vec<[bool; 4]>,
    prizes: Vec<(usize, usize)>,
}

impl MazeData {
    fn idx(&self, x: usize, y: usize) -> usize { y * self.width + x }
    fn wall(&self, x: usize, y: usize, d: Dir) -> bool { self.walls[self.idx(x, y)][d as usize] }
    fn end(&self) -> (usize, usize) { (self.width - 1, self.height - 1) }
}

fn build_maze(width: usize, height: usize, rng: &mut StdRng) -> MazeData {
    let mut walls = vec![[true; 4]; width * height];
    let mut visited = vec![false; width * height];
    let idx = |x: usize, y: usize| y * width + x;

    let mut stack = vec![(0usize, 0usize)];
    visited[0] = true;

    while let Some(&(cx, cy)) = stack.last() {
        let candidates: Vec<(usize, usize, Dir)> = Dir::ALL.iter()
            .filter_map(|&d| {
                let (dx, dy) = d.delta();
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && ny >= 0 && nx < width as isize && ny < height as isize {
                    let (nx, ny) = (nx as usize, ny as usize);
                    if !visited[idx(nx, ny)] { return Some((nx, ny, d)); }
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

// ── Global maze (replaceable between runs) ───────────────────────────────────

static CURRENT_MAZE: OnceLock<RwLock<Option<Arc<MazeData>>>> = OnceLock::new();

fn maze_lock() -> &'static RwLock<Option<Arc<MazeData>>> {
    CURRENT_MAZE.get_or_init(|| RwLock::new(None))
}

fn set_maze(maze: MazeData) {
    *maze_lock().write().unwrap() = Some(Arc::new(maze));
}

fn get_maze() -> Arc<MazeData> {
    maze_lock().read().unwrap().clone().expect("maze not set")
}

// ── Path simulation ───────────────────────────────────────────────────────────

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
            let unused = (decisions.len() - dec_i) as f64;
            let score = max_dist + SOLVED_BONUS
                + prizes_collected as f64 * PRIZE_VALUE
                - path.len() as f64 * STEP_PENALTY
                - unused * EXTRA_DECISION_PENALTY;
            return (score, path, true);
        }

        for (i, &(px, py)) in maze.prizes.iter().enumerate() {
            if px == x && py == y && !collected[i] {
                collected[i] = true;
                prizes_collected += 1;
            }
        }

        let choices: Vec<Dir> = Dir::ALL.iter().copied()
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
            if dec_i >= decisions.len() { break; }
            let gene = decisions[dec_i].0;
            dec_i += 1;

            if gene == 3 {
                back_dir.unwrap_or(choices[0])
            } else {
                let order = [facing.left(), facing, facing.right()];
                let sorted: Vec<Dir> = order.iter().copied()
                    .filter(|d| choices.contains(d))
                    .collect();
                match sorted.len() {
                    0 => choices[0],
                    1 => sorted[0],
                    2 => if gene == 0 { sorted[0] } else { sorted[1] },
                    _ => sorted[gene as usize],
                }
            }
        };

        facing = chosen;
        let (dx, dy) = chosen.delta();
        pos = ((x as isize + dx) as usize, (y as isize + dy) as usize);
        back_dir = Some(chosen.opposite());
        path.push(pos);

        if path.len() > maze.width * maze.height + 10 { break; }
    }

    let (x, y) = pos;
    let unused = (decisions.len() - dec_i) as f64;
    let manhattan = ((x as isize - end_x as isize).abs() + (y as isize - end_y as isize).abs()) as f64;
    let score = (max_dist - manhattan)
        + prizes_collected as f64 * PRIZE_VALUE
        - path.len() as f64 * STEP_PENALTY
        - unused * EXTRA_DECISION_PENALTY;
    (score, path, false)
}

// ── Gene / Genome ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Decision(u8);

impl Generate for Decision {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        Decision(rng.gen_range(0u8..4))
    }
}

impl Mutate for Decision {
    fn mutate(&self, _c: &MutationConfig, seed: [u8; 32]) -> Self { Decision::generate(seed) }
}

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
    fn get_fitness(&self) -> Option<f64> { self.0.get_fitness() }
}

impl Fitness for MazePath {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        if self.0.get_fitness().is_some() { return self.0.get_fitness(); }
        let maze = get_maze();
        let (score, _, _) = simulate(&maze, self.0.get_data());
        self.0.set_fitness(Some(score));
        Some(score)
    }
}

// ── API types ─────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct CellWalls { n: bool, e: bool, s: bool, w: bool }

#[derive(Serialize, Clone)]
struct ApiState {
    width: usize,
    height: usize,
    generation: usize,
    total_generations: usize,
    score: f64,
    solved: bool,
    prizes: Vec<[usize; 2]>,
    path: Vec<[usize; 2]>,
    cells: Vec<CellWalls>,
    maze_id: u64,
}

// ── GA background thread ──────────────────────────────────────────────────────

fn ga_thread(shared: Arc<RwLock<ApiState>>) {
    let mut maze_id = 0u64;
    loop {
        let seed: [u8; 32] = rand::thread_rng().gen();
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let maze = build_maze(MAZE_W, MAZE_H, &mut rng);

        let cells: Vec<CellWalls> = (0..MAZE_H)
            .flat_map(|y| (0..MAZE_W).map(move |x| (x, y)))
            .map(|(x, y)| CellWalls {
                n: maze.wall(x, y, Dir::N),
                e: maze.wall(x, y, Dir::E),
                s: maze.wall(x, y, Dir::S),
                w: maze.wall(x, y, Dir::W),
            })
            .collect();

        let prizes: Vec<[usize; 2]> = maze.prizes.iter().map(|&(x, y)| [x, y]).collect();
        set_maze(maze);
        maze_id += 1;

        let config = PopulationConfig {
            pop_size: 300,
            crossover_count: 80,
            mutate_count: 80,
            elitism_count: 20,
            mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
            seed: rng.gen(),
            preseeded_population: vec![],
        };
        let mut population: Population<MazePath> = Population::new(config);

        for gen in 0..GENS_PER_MAZE {
            population.tick();
            std::thread::sleep(std::time::Duration::from_secs(1));

            let best = population.get_best_member();
            let maze = get_maze();
            let (score, path, solved) = simulate(&maze, best.0.get_data());
            let path_api: Vec<[usize; 2]> = path.iter().map(|&(x, y)| [x, y]).collect();

            *shared.write().unwrap() = ApiState {
                width: MAZE_W,
                height: MAZE_H,
                generation: gen + 1,
                total_generations: GENS_PER_MAZE,
                score,
                solved,
                prizes: prizes.clone(),
                path: path_api,
                cells: cells.clone(),
                maze_id,
            };
        }
    }
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

async fn handle_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn handle_state(State(shared): State<Arc<RwLock<ApiState>>>) -> Json<ApiState> {
    Json(shared.read().unwrap().clone())
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let initial = ApiState {
        width: MAZE_W,
        height: MAZE_H,
        generation: 0,
        total_generations: GENS_PER_MAZE,
        score: 0.0,
        solved: false,
        prizes: vec![],
        path: vec![],
        cells: vec![],
        maze_id: 0,
    };
    let shared = Arc::new(RwLock::new(initial));
    let shared_ga = shared.clone();

    std::thread::spawn(move || ga_thread(shared_ga));

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/api/state", get(handle_state))
        .with_state(shared);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("Serving on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

mod maze_html;
use maze_html::INDEX_HTML;
