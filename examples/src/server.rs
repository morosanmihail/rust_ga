#[allow(dead_code)]
mod swarm;
#[allow(dead_code)]
mod tower_defence;

use axum::{response::Html, routing::get, Router};

// ─── Swarm ────────────────────────────────────────────────────────────────────

mod swarm_impl {
    use std::sync::{Arc, RwLock};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use axum::{extract::State, response::Html, routing::get, Json, Router};
    use ga::population::{MutationConfig, Population, PopulationConfig};
    use rand::Rng;
    use serde::Serialize;
    use crate::swarm::evolve::{update_snapshot, SteeringGenome};
    use crate::swarm::sim::{SwarmSim, WORLD_H, WORLD_W};

    const GA_GENS_PER_TICK: usize = 20;
    const MAX_SIM_TICKS: u64 = 300;
    const TICK_SLEEP_MS: u64 = 50;

    #[derive(Serialize, Clone)]
    struct AgentSnap { x: f64, y: f64, vx: f64, vy: f64, done: bool }

    #[derive(Serialize, Clone, Default)]
    struct ApiState {
        tick: u64, ga_gens: u64, run: u64,
        agents: Vec<AgentSnap>,
        obstacles: Vec<[f64; 3]>,
        goal: [f64; 4],
        world_w: f64, world_h: f64,
        weights: [f32; 4],
        agents_at_goal: usize, total_agents: usize, finished: bool,
    }

    struct Shared { state: RwLock<ApiState>, last_hit: AtomicU64 }

    fn now_secs() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    fn make_pop() -> Population<SteeringGenome> {
        Population::new(PopulationConfig {
            seed: rand::thread_rng().gen(),
            pop_size: 40, elitism_count: 4, mutate_count: 16, crossover_count: 10,
            mutation_config: MutationConfig { gene_mutation_chance: 0.5 },
            preseeded_population: vec![],
        })
    }

    fn ga_thread(shared: Arc<Shared>) {
        let mut run = 0u64;
        let mut total_ga_gens = 0u64;
        loop {
            run += 1;
            let mut sim = SwarmSim::new_random(rand::thread_rng().gen());
            update_snapshot(&sim);
            let obstacles: Vec<[f64; 3]> = sim.obstacles.iter()
                .map(|o| [o.pos.0, o.pos.1, o.radius]).collect();
            let mut pop = make_pop();
            loop {
                if now_secs().saturating_sub(shared.last_hit.load(Ordering::Relaxed)) > 30 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
                for _ in 0..GA_GENS_PER_TICK { pop.tick_parallel(); total_ga_gens += 1; }
                let best = pop.get_best_member().clone();
                sim.step(&best.to_weights());
                update_snapshot(&sim);
                let agents: Vec<AgentSnap> = sim.agents.iter()
                    .map(|a| AgentSnap {
                        x: a.pos.0, y: a.pos.1, vx: a.vel.0, vy: a.vel.1, done: a.reached_goal,
                    })
                    .collect();
                let finished = sim.agents_at_goal() == sim.agents.len() || sim.tick >= MAX_SIM_TICKS;
                *shared.state.write().unwrap() = ApiState {
                    tick: sim.tick, ga_gens: total_ga_gens, run, agents,
                    obstacles: obstacles.clone(),
                    goal: [sim.goal.x, sim.goal.y, sim.goal.w, sim.goal.h],
                    world_w: WORLD_W, world_h: WORLD_H,
                    weights: best.weights,
                    agents_at_goal: sim.agents_at_goal(),
                    total_agents: sim.agents.len(), finished,
                };
                if finished {
                    std::thread::sleep(std::time::Duration::from_secs(4));
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(TICK_SLEEP_MS));
            }
        }
    }

    async fn handle_index() -> Html<&'static str> { Html(include_str!("html/swarm.html")) }

    async fn handle_state(State(s): State<Arc<Shared>>) -> Json<ApiState> {
        s.last_hit.store(now_secs(), Ordering::Relaxed);
        Json(s.state.read().unwrap().clone())
    }

    pub fn make_router() -> Router {
        let shared = Arc::new(Shared {
            state: RwLock::new(ApiState::default()),
            last_hit: AtomicU64::new(0),
        });
        let s = shared.clone();
        std::thread::spawn(move || ga_thread(s));
        Router::new()
            .route("/", get(handle_index))
            .route("/api/state", get(handle_state))
            .with_state(shared)
    }
}

// ─── Maze ─────────────────────────────────────────────────────────────────────

mod maze_impl {
    use std::sync::{Arc, OnceLock, RwLock};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Dir { N = 0, E = 1, S = 2, W = 3 }

    impl Dir {
        const ALL: [Dir; 4] = [Dir::N, Dir::E, Dir::S, Dir::W];
        fn opposite(self) -> Dir { match self { Dir::N=>Dir::S, Dir::E=>Dir::W, Dir::S=>Dir::N, Dir::W=>Dir::E } }
        fn left(self) -> Dir    { match self { Dir::N=>Dir::W, Dir::E=>Dir::N, Dir::S=>Dir::E, Dir::W=>Dir::S } }
        fn right(self) -> Dir   { match self { Dir::N=>Dir::E, Dir::E=>Dir::S, Dir::S=>Dir::W, Dir::W=>Dir::N } }
        fn delta(self) -> (isize, isize) { match self { Dir::N=>(0,-1), Dir::E=>(1,0), Dir::S=>(0,1), Dir::W=>(-1,0) } }
    }

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

    #[derive(Serialize, Clone)]
    struct CellWalls { n: bool, e: bool, s: bool, w: bool }

    #[derive(Serialize, Clone)]
    struct ApiState {
        width: usize, height: usize,
        generation: usize, total_generations: usize,
        score: f64, solved: bool,
        prizes: Vec<[usize; 2]>,
        path: Vec<[usize; 2]>,
        cells: Vec<CellWalls>,
        maze_id: u64,
    }

    struct Shared { state: RwLock<ApiState>, last_hit: AtomicU64 }

    fn now_secs() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    fn ga_thread(shared: Arc<Shared>) {
        let mut maze_id = 0u64;
        loop {
            let seed: [u8; 32] = rand::thread_rng().gen();
            let mut rng: StdRng = SeedableRng::from_seed(seed);
            let maze = build_maze(MAZE_W, MAZE_H, &mut rng);
            let cells: Vec<CellWalls> = (0..MAZE_H)
                .flat_map(|y| (0..MAZE_W).map(move |x| (x, y)))
                .map(|(x, y)| CellWalls {
                    n: maze.wall(x, y, Dir::N), e: maze.wall(x, y, Dir::E),
                    s: maze.wall(x, y, Dir::S), w: maze.wall(x, y, Dir::W),
                })
                .collect();
            let prizes: Vec<[usize; 2]> = maze.prizes.iter().map(|&(x, y)| [x, y]).collect();
            set_maze(maze);
            maze_id += 1;
            let config = PopulationConfig {
                pop_size: 300, crossover_count: 80, mutate_count: 80, elitism_count: 20,
                mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
                seed: rng.gen(), preseeded_population: vec![],
            };
            let mut population: Population<MazePath> = Population::new(config);
            let mut gen = 0;
            while gen < GENS_PER_MAZE {
                if now_secs().saturating_sub(shared.last_hit.load(Ordering::Relaxed)) > 30 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
                population.tick();
                std::thread::sleep(std::time::Duration::from_secs(1));
                gen += 1;
                let best = population.get_best_member();
                let maze = get_maze();
                let (score, path, solved) = simulate(&maze, best.0.get_data());
                let path_api: Vec<[usize; 2]> = path.iter().map(|&(x, y)| [x, y]).collect();
                *shared.state.write().unwrap() = ApiState {
                    width: MAZE_W, height: MAZE_H,
                    generation: gen, total_generations: GENS_PER_MAZE,
                    score, solved,
                    prizes: prizes.clone(),
                    path: path_api,
                    cells: cells.clone(),
                    maze_id,
                };
            }
        }
    }

    async fn handle_index() -> Html<&'static str> { Html(include_str!("html/maze.html")) }

    async fn handle_state(State(shared): State<Arc<Shared>>) -> Json<ApiState> {
        shared.last_hit.store(now_secs(), Ordering::Relaxed);
        Json(shared.state.read().unwrap().clone())
    }

    pub fn make_router() -> Router {
        let shared = Arc::new(Shared {
            state: RwLock::new(ApiState {
                width: MAZE_W, height: MAZE_H,
                generation: 0, total_generations: GENS_PER_MAZE,
                score: 0.0, solved: false,
                prizes: vec![], path: vec![], cells: vec![], maze_id: 0,
            }),
            last_hit: AtomicU64::new(0),
        });
        let s = shared.clone();
        std::thread::spawn(move || ga_thread(s));
        Router::new()
            .route("/", get(handle_index))
            .route("/api/state", get(handle_state))
            .with_state(shared)
    }
}

// ─── Tower Defence ───────────────────────────────────────────────────────────

mod td_impl {
    use std::sync::{Arc, RwLock};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use axum::{extract::State, response::Html, routing::get, Json, Router};
    use ga::population::{MutationConfig, Population, PopulationConfig};
    use ga::traits::FitnessRetrieve;
    use rand::Rng;
    use serde::Serialize;
    use crate::tower_defence::{
        builder::{Builder, Instruction},
        evolve::{eval_config, make_eval_map, max_fitness, set_map_seed, BuilderGenome, BUILDER_HP, BUILDER_START},
        map::{Structure, Terrain},
        simulation::Simulation,
    };

    #[derive(Serialize, Clone)]
    struct EnemySnap { x: usize, y: usize, hp: u32 }

    #[derive(Serialize, Clone)]
    struct StructSnap { x: usize, y: usize, kind: u8, hp: u32 }

    #[derive(Serialize, Clone)]
    struct TickSnap {
        tick: u64,
        builder_x: usize, builder_y: usize, builder_hp: u32,
        builder_instr_idx: Option<usize>,
        enemies_killed: u32,
        enemies: Vec<EnemySnap>,
        structures: Vec<StructSnap>,
    }

    #[derive(Serialize, Clone, Default)]
    struct ApiState {
        width: usize, height: usize,
        generation: usize, fitness: f64, max_fitness: f64,
        terrain: Vec<u8>,
        spawn_points: Vec<[usize; 2]>,
        builder_max_hp: u32, enemy_max_hp: u32,
        builder_instrs: Vec<String>,
        ticks: Vec<TickSnap>,
        survived: bool, ready: bool,
    }

    fn snap_tick(sim: &Simulation) -> TickSnap {
        let enemies = sim.enemies.iter()
            .map(|e| EnemySnap { x: e.x, y: e.y, hp: e.hp })
            .collect();
        let mut structures = Vec::new();
        for y in 0..sim.map.height {
            for x in 0..sim.map.width {
                if let Some(s) = &sim.map.get(x, y).structure {
                    let (kind, hp) = match s {
                        Structure::Tower { hp } => (1u8, *hp),
                        Structure::Wall  { hp } => (2u8, *hp),
                        Structure::Bridge       => (3u8, 0u32),
                    };
                    structures.push(StructSnap { x, y, kind, hp });
                }
            }
        }
        TickSnap {
            tick: sim.tick,
            builder_x: sim.builder.x, builder_y: sim.builder.y,
            builder_hp: sim.builder.hp,
            builder_instr_idx: sim.builder.current_instr_idx,
            enemies_killed: sim.enemies_killed,
            enemies, structures,
        }
    }

    fn fmt_instr(i: &Instruction) -> String {
        match i {
            Instruction::Move(x, y)        => format!("Move {},{}", x, y),
            Instruction::BuildTower(x, y)  => format!("Tower {},{}", x, y),
            Instruction::BuildWall(x, y)   => format!("Wall {},{}", x, y),
            Instruction::BuildBridge(x, y) => format!("Bridge {},{}", x, y),
        }
    }

    fn record_sim(genome: &BuilderGenome) -> (Vec<String>, Vec<TickSnap>, bool) {
        let instrs = genome.to_instructions();
        let builder_instrs: Vec<String> = instrs.iter().map(fmt_instr).collect();
        let mut builder = Builder::new(BUILDER_START.0, BUILDER_START.1, BUILDER_HP);
        for instr in instrs { builder.push(instr); }
        let mut sim = Simulation::new(make_eval_map(), builder, eval_config());
        let mut ticks = vec![snap_tick(&sim)];
        for _ in 0..300 {
            sim.tick();
            ticks.push(snap_tick(&sim));
            if sim.is_game_over() { break; }
        }
        let survived = !sim.is_game_over();
        (builder_instrs, ticks, survived)
    }

    fn make_pop_config() -> PopulationConfig<BuilderGenome> {
        PopulationConfig {
            pop_size: 80, crossover_count: 25, mutate_count: 25, elitism_count: 5,
            mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
            seed: rand::thread_rng().gen(), preseeded_population: vec![],
        }
    }

    fn make_map_info() -> (usize, usize, Vec<u8>, Vec<[usize; 2]>) {
        let map = make_eval_map();
        let width = map.width;
        let height = map.height;
        let spawn_points: Vec<[usize; 2]> = map.spawn_points.iter().map(|&(x, y)| [x, y]).collect();
        let mut terrain = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                terrain.push(match map.get(x, y).terrain {
                    Terrain::Plain => 0,
                    Terrain::Rock  => 1,
                    Terrain::Water => 2,
                });
            }
        }
        (width, height, terrain, spawn_points)
    }

    struct Shared { state: RwLock<ApiState>, last_hit: AtomicU64 }

    fn now_secs() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    fn ga_thread(shared: Arc<Shared>) {
        let mut rng = rand::thread_rng();
        let cfg = eval_config();
        set_map_seed(rng.gen());
        let (mut width, mut height, mut terrain, mut spawn_points) = make_map_info();
        let mut mf = max_fitness();
        let mut pop: Population<BuilderGenome> = Population::new(make_pop_config());
        let mut gen = 0usize;
        loop {
            if now_secs().saturating_sub(shared.last_hit.load(Ordering::Relaxed)) > 30 {
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
            std::thread::sleep(std::time::Duration::from_millis(2500));
            pop.tick_parallel();
            gen += 1;
            let best = pop.get_best_member().clone();
            let fitness = best.get_fitness().unwrap_or(0.0);
            let (builder_instrs, ticks, survived) = record_sim(&best);
            *shared.state.write().unwrap() = ApiState {
                width, height, generation: gen, fitness, max_fitness: mf,
                terrain: terrain.clone(), spawn_points: spawn_points.clone(),
                builder_max_hp: BUILDER_HP, enemy_max_hp: cfg.enemy_hp,
                builder_instrs, ticks, survived, ready: true,
            };
            if fitness >= mf || gen >= 150 {
                if fitness >= mf { std::thread::sleep(std::time::Duration::from_secs(60)); }
                set_map_seed(rng.gen());
                (width, height, terrain, spawn_points) = make_map_info();
                mf = max_fitness();
                pop = Population::new(make_pop_config());
                gen = 0;
            }
        }
    }

    async fn handle_index() -> Html<&'static str> { Html(include_str!("html/tower_defence.html")) }

    async fn handle_state(State(s): State<Arc<Shared>>) -> Json<ApiState> {
        s.last_hit.store(now_secs(), Ordering::Relaxed);
        Json(s.state.read().unwrap().clone())
    }

    pub fn make_router() -> Router {
        let shared = Arc::new(Shared {
            state: RwLock::new(ApiState::default()),
            last_hit: AtomicU64::new(0),
        });
        let s = shared.clone();
        std::thread::spawn(move || ga_thread(s));
        Router::new()
            .route("/", get(handle_index))
            .route("/api/state", get(handle_state))
            .with_state(shared)
    }
}

// ─── Root index ───────────────────────────────────────────────────────────────

async fn handle_root() -> Html<&'static str> { Html(include_str!("html/root.html")) }

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(handle_root))
        .nest("/swarm", swarm_impl::make_router())
        .nest("/maze", maze_impl::make_router())
        .nest("/tower-defence", td_impl::make_router());

    println!("GA examples server: http://localhost:3000");
    println!("  /swarm          Swarm steering");
    println!("  /maze           Maze solver");
    println!("  /tower-defence  Tower defence");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
