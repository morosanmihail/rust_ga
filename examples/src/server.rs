#[allow(dead_code)]
mod swarm;
#[allow(dead_code)]
mod tower_defence;
mod maze_html;

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

    async fn handle_index() -> Html<&'static str> { Html(HTML) }

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

    pub const HTML: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swarm — Per-tick GA</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:'Courier New',monospace;
  display:flex;flex-direction:column;align-items:center;
  min-height:100vh;padding:20px 12px;gap:14px;
}
nav{display:flex;gap:12px;font-size:11px}
nav a{color:#8b949e;text-decoration:none;padding:4px 10px;border:1px solid #30363d;border-radius:6px}
nav a:hover{color:#c9d1d9;border-color:#8b949e}
nav a.active{color:#58a6ff;border-color:#58a6ff}
h1{color:#58a6ff;font-size:1.2rem;letter-spacing:4px;text-transform:uppercase}
#stats{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.stat{
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:7px 14px;text-align:center;min-width:88px;
}
.stat-label{font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.stat-value{font-size:17px;font-weight:bold;color:#58a6ff;margin-top:2px}
.green{color:#3fb950}
canvas{display:block;border:1px solid #21262d;border-radius:6px;max-width:100%;image-rendering:pixelated}
#weights-row{
  display:flex;align-items:flex-end;gap:4px;
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:10px 16px 6px;
}
.wbar-wrap{display:flex;flex-direction:column;align-items:center;gap:3px;min-width:52px}
.wbar-label{font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.wbar-val{font-size:11px;color:#c9d1d9}
.wbar-track{
  width:40px;height:50px;background:#0d1117;border-radius:4px;
  display:flex;align-items:flex-end;overflow:hidden;
}
.wbar-fill{width:100%;border-radius:4px 4px 0 0;transition:height 0.15s ease}
#legend{display:flex;gap:14px;font-size:11px;color:#8b949e;flex-wrap:wrap;justify-content:center}
.leg{display:flex;align-items:center;gap:5px}
.lc{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.ll{width:18px;height:3px;flex-shrink:0}
#footer{font-size:10px;color:#484f58;letter-spacing:1px;text-align:center}
#loading{font-size:14px;color:#8b949e;padding:40px}
</style>
</head>
<body>
<nav>
  <a href="/">Home</a>
  <a href="/swarm" class="active">Swarm</a>
  <a href="/maze">Maze</a>
  <a href="/tower-defence">Tower Defence</a>
</nav>
<h1>Swarm Steering — Per-tick Genetic Algorithm</h1>

<div id="stats">
  <div class="stat"><div class="stat-label">Run</div><div class="stat-value" id="s-run">—</div></div>
  <div class="stat"><div class="stat-label">Sim Tick</div><div class="stat-value" id="s-tick">—</div></div>
  <div class="stat"><div class="stat-label">GA Gens</div><div class="stat-value" id="s-gens">—</div></div>
  <div class="stat"><div class="stat-label">At Goal</div><div class="stat-value green" id="s-goal">—</div></div>
</div>

<canvas id="canvas"></canvas>

<div id="weights-row">
  <div class="wbar-wrap">
    <div class="wbar-track"><div class="wbar-fill" id="w0" style="background:#58a6ff"></div></div>
    <div class="wbar-val" id="wv0">—</div>
    <div class="wbar-label">Goal</div>
  </div>
  <div class="wbar-wrap">
    <div class="wbar-track"><div class="wbar-fill" id="w1" style="background:#f85149"></div></div>
    <div class="wbar-val" id="wv1">—</div>
    <div class="wbar-label">Obstacle</div>
  </div>
  <div class="wbar-wrap">
    <div class="wbar-track"><div class="wbar-fill" id="w2" style="background:#3fb950"></div></div>
    <div class="wbar-val" id="wv2">—</div>
    <div class="wbar-label">Align</div>
  </div>
  <div class="wbar-wrap">
    <div class="wbar-track"><div class="wbar-fill" id="w3" style="background:#e3b341"></div></div>
    <div class="wbar-val" id="wv3">—</div>
    <div class="wbar-label">Separate</div>
  </div>
</div>

<div id="legend">
  <div class="leg"><div class="lc" style="background:#58a6ff;border:1.5px solid #1f6feb"></div>Agent (moving)</div>
  <div class="leg"><div class="lc" style="background:#3fb950;border:1.5px solid #2ea043"></div>Agent (done)</div>
  <div class="leg"><div class="lc" style="background:#b94040;border:1.5px solid #ff5555"></div>Obstacle</div>
  <div class="leg"><div class="ll" style="background:#3fb950;height:12px;border-radius:2px"></div>Goal zone</div>
</div>

<div id="footer">Evolves 4 steering weights [goal_pull, obstacle_push, neighbor_align, neighbor_separate] for a boid swarm — GA runs 20 generations on current snapshot each tick, best weights applied to advance sim one step.</div>
<div id="loading">Starting…</div>

<script>
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const loading = document.getElementById('loading');

let state = null;
const SCALE = 14;
const W_COLORS = ['#58a6ff','#f85149','#3fb950','#e3b341'];
const MAX_W = 5.0;

function draw() {
  if (!state || !state.total_agents) return;
  loading.style.display = 'none';

  const CW = Math.round(state.world_w * SCALE);
  const CH = Math.round(state.world_h * SCALE);
  if (canvas.width !== CW || canvas.height !== CH) {
    canvas.width  = CW;
    canvas.height = CH;
  }

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, CW, CH);

  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= state.world_w; x++) {
    ctx.beginPath(); ctx.moveTo(x*SCALE,0); ctx.lineTo(x*SCALE,CH); ctx.stroke();
  }
  for (let y = 0; y <= state.world_h; y++) {
    ctx.beginPath(); ctx.moveTo(0,y*SCALE); ctx.lineTo(CW,y*SCALE); ctx.stroke();
  }

  for (const [ox, oy, r] of state.obstacles) {
    const px = ox * SCALE, py = oy * SCALE, pr = r * SCALE;
    const g = ctx.createRadialGradient(px, py, pr*0.2, px, py, pr);
    g.addColorStop(0, 'rgba(200,50,50,0.85)');
    g.addColorStop(1, 'rgba(120,20,20,0.55)');
    ctx.fillStyle = g;
    ctx.strokeStyle = '#ff5555';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(px, py, pr, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  }

  const [grx, gry, grw, grh] = state.goal;
  const gpx = grx * SCALE, gpy = gry * SCALE;
  const gpw = grw * SCALE, gph = grh * SCALE;
  const glAlpha = 0.18 + 0.10 * Math.sin(Date.now() * 0.003);
  ctx.fillStyle = `rgba(63,185,80,${glAlpha})`;
  ctx.fillRect(gpx, gpy, gpw, gph);
  const borderAlpha = 0.7 + 0.3 * Math.sin(Date.now() * 0.003);
  ctx.strokeStyle = `rgba(63,185,80,${borderAlpha})`;
  ctx.lineWidth = 2;
  ctx.strokeRect(gpx, gpy, gpw, gph);
  ctx.fillStyle = '#3fb950';
  ctx.font = `bold 10px 'Courier New', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('GOAL', gpx + gpw / 2, gpy + gph / 2);

  for (const a of state.agents) {
    const ax = a.x * SCALE, ay = a.y * SCALE;
    if (!a.done) {
      const speed = Math.hypot(a.vx, a.vy);
      if (speed > 0.05) {
        const arrowLen = speed / 0.7 * SCALE * 0.6;
        ctx.strokeStyle = 'rgba(88,166,255,0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax + a.vx/speed * arrowLen, ay + a.vy/speed * arrowLen);
        ctx.stroke();
      }
    }
    const r = a.done ? 5 : 4;
    ctx.fillStyle   = a.done ? '#3fb950' : '#58a6ff';
    ctx.strokeStyle = a.done ? '#2ea043' : '#1f6feb';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(ax, ay, r, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  }

  if (state.finished) {
    const allDone = state.agents_at_goal === state.total_agents;
    ctx.fillStyle = allDone ? 'rgba(63,185,80,0.12)' : 'rgba(248,81,73,0.12)';
    ctx.fillRect(0, 0, CW, CH);
    ctx.fillStyle = allDone ? '#3fb950' : '#f85149';
    ctx.font = `bold ${SCALE*1.2}px 'Courier New',monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const msg = allDone
      ? `All ${state.total_agents} agents reached goal — tick ${state.tick}`
      : `${state.agents_at_goal}/${state.total_agents} reached goal — timed out at tick ${state.tick}`;
    ctx.fillText(msg, CW/2, CH/2);
  }

  document.getElementById('s-run').textContent  = state.run;
  document.getElementById('s-tick').textContent = state.tick;
  document.getElementById('s-gens').textContent = state.ga_gens.toLocaleString();
  const goalEl = document.getElementById('s-goal');
  goalEl.textContent = `${state.agents_at_goal}/${state.total_agents}`;
  goalEl.className = 'stat-value ' + (state.agents_at_goal === state.total_agents ? 'green' : '');

  for (let i = 0; i < 4; i++) {
    const v = state.weights[i];
    const pct = Math.min(1, v / MAX_W);
    document.getElementById(`w${i}`).style.height = (pct * 50) + 'px';
    document.getElementById(`wv${i}`).textContent = v.toFixed(2);
  }
}

function poll() {
  fetch('/swarm/api/state')
    .then(r => r.json())
    .then(d => { state = d; })
    .catch(() => {})
    .finally(() => setTimeout(poll, 100));
}

function loop() {
  draw();
  requestAnimationFrame(loop);
}

poll();
loop();
</script>
</body>
</html>
"####;
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

    async fn handle_index() -> Html<&'static str> { Html(crate::maze_html::INDEX_HTML) }

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

    async fn handle_index() -> Html<&'static str> { Html(HTML) }

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

    pub const HTML: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tower Defence — GA</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:'Courier New',monospace;
  display:flex;flex-direction:column;align-items:center;
  min-height:100vh;padding:20px 12px;gap:14px;
}
nav{display:flex;gap:12px;font-size:11px}
nav a{color:#8b949e;text-decoration:none;padding:4px 10px;border:1px solid #30363d;border-radius:6px}
nav a:hover{color:#c9d1d9;border-color:#8b949e}
nav a.active{color:#58a6ff;border-color:#58a6ff}
h1{color:#58a6ff;font-size:1.25rem;letter-spacing:4px;text-transform:uppercase}
#stats{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.stat{
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:7px 14px;text-align:center;min-width:88px;
}
.stat-label{font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.stat-value{font-size:17px;font-weight:bold;color:#58a6ff;margin-top:2px}
.green{color:#3fb950}.gold{color:#e3b341}.red{color:#f85149}
#map-row{display:flex;gap:12px;align-items:flex-start}
#canvas-wrap{position:relative;flex-shrink:0}
canvas{display:block;border:1px solid #21262d;border-radius:6px;
  image-rendering:pixelated;max-width:100%}
#instr-panel{
  width:160px;flex-shrink:0;
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  overflow-y:auto;
  font-size:12px;font-family:'Courier New',monospace;
}
#instr-panel h2{
  font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;
  padding:8px 10px 4px;border-bottom:1px solid #21262d;margin:0;
  position:sticky;top:0;background:#161b22;z-index:1;
}
.instr-item{
  padding:4px 10px;border-bottom:1px solid #21262d22;
  color:#8b949e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  display:flex;gap:6px;align-items:center;cursor:default;
}
.instr-item.done{color:#3fb95066;text-decoration:line-through}
.instr-item.active{
  background:#1f3a1f;color:#3fb950;
  border-left:3px solid #3fb950;padding-left:7px;font-weight:bold;
}
.instr-item .idx{color:#484f58;font-size:10px;min-width:16px;text-align:right;flex-shrink:0}
#legend{
  display:flex;gap:12px;font-size:11px;color:#8b949e;
  flex-wrap:wrap;justify-content:center;
}
.leg{display:flex;align-items:center;gap:5px}
.lb{width:14px;height:14px;border-radius:3px;flex-shrink:0}
.lc{width:14px;height:14px;border-radius:50%;flex-shrink:0}
#footer{font-size:10px;color:#484f58;letter-spacing:1px;margin-top:2px;text-align:center}
#loading{font-size:14px;color:#8b949e;padding:40px}
</style>
</head>
<body>
<nav>
  <a href="/">Home</a>
  <a href="/swarm">Swarm</a>
  <a href="/maze">Maze</a>
  <a href="/tower-defence" class="active">Tower Defence</a>
</nav>
<h1>Tower Defence — Genetic Algorithm</h1>

<div id="stats">
  <div class="stat"><div class="stat-label">Generation</div><div class="stat-value" id="s-gen">—</div></div>
  <div class="stat"><div class="stat-label">Fitness</div><div class="stat-value" id="s-fit">—</div></div>
  <div class="stat"><div class="stat-label">Tick</div><div class="stat-value" id="s-tick">—</div></div>
  <div class="stat"><div class="stat-label">Builder HP</div><div class="stat-value" id="s-bhp">—</div></div>
  <div class="stat"><div class="stat-label">Enemies</div><div class="stat-value" id="s-ene">—</div></div>
  <div class="stat"><div class="stat-label">Killed</div><div class="stat-value green" id="s-kill">—</div></div>
</div>

<div id="map-row">
  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
    <div id="loading">Waiting for first GA generation…</div>
  </div>
  <div id="instr-panel">
    <h2>Instructions</h2>
    <div id="instr-list"></div>
  </div>
</div>

<div id="legend">
  <div class="leg"><div class="lb" style="background:#2d5a3d;outline:1px solid #3d7a53"></div>Plain</div>
  <div class="leg"><div class="lb" style="background:#4a4a50"></div>Rock</div>
  <div class="leg"><div class="lb" style="background:#0a3d62;outline:1px solid #1a5a8a"></div>Water</div>
  <div class="leg"><div class="lb" style="background:#708090;outline:2px solid #4a6070"></div>Tower</div>
  <div class="leg"><div class="lb" style="background:#8b6914;border-radius:2px"></div>Wall</div>
  <div class="leg"><div class="lb" style="background:#c8a05a;border-radius:2px"></div>Bridge</div>
  <div class="leg"><div class="lc" style="background:#f0c040;outline:2px solid #c08000"></div>Builder</div>
  <div class="leg"><div class="lc" style="background:#c0392b;outline:1px solid #922b21"></div>Enemy</div>
  <div class="leg"><div class="lb" style="background:transparent;outline:2px dashed #f0a500;border-radius:0"></div>Spawn</div>
</div>

<div id="footer">Evolves a sequence of builder instructions (Move/BuildTower/BuildWall/BuildBridge) — best genome fed as instruction queue for a 300-tick simulation, fitness = ticks survived (+ HP if alive at end).</div>

<script>
const canvas    = document.getElementById('canvas');
const ctx       = canvas.getContext('2d');
const loading   = document.getElementById('loading');
const instrList = document.getElementById('instr-list');
const instrPanel= document.getElementById('instr-panel');
const CELL      = 40;

let state = null, tickIdx = 0, fetching = false, lastTickMs = 0;
let pulseT = 0;

const cx  = x  => x  * CELL + CELL / 2;
const cy  = y  => y  * CELL + CELL / 2;
const bx  = x  => x  * CELL;
const by  = y  => y  * CELL;

function hpBar(centerX, top, totalW, hp, maxHp, fillColor) {
  if (maxHp <= 0) return;
  const w = totalW * 0.82, h = 3.5;
  const x0 = centerX - w / 2;
  ctx.fillStyle = '#1c2128';
  ctx.beginPath(); ctx.roundRect(x0, top, w, h, 2); ctx.fill();
  ctx.fillStyle = fillColor;
  ctx.beginPath(); ctx.roundRect(x0, top, w * Math.max(0, hp / maxHp), h, 2); ctx.fill();
}

const T_FILL = ['#2d5a3d','#4a4a50','#0a3d62'];

function ch(gx, gy, n) {
  let v = Math.imul(gx * 1664525 + n * 6364136, gy * 1013904223 + n * 22695477);
  v ^= v >>> 13; v = Math.imul(v ^ 0x9e3779b9, 0x6c62272e);
  v ^= v >>> 15;
  return ((v >>> 0) & 0xffff) / 0x10000;
}

function drawRock(px, py, gx, gy) {
  ctx.fillStyle = '#3a3a40';
  ctx.fillRect(px, py, CELL, CELL);
  const shades = ['#4a4a52','#424249','#525259','#3d3d44','#565660','#46464e'];
  for (let f = 0; f < 5; f++) {
    const x0 = px + ch(gx,gy,f*7  ) * CELL;
    const y0 = py + ch(gx,gy,f*7+1) * CELL;
    const x1 = px + ch(gx,gy,f*7+2) * CELL;
    const y1 = py + ch(gx,gy,f*7+3) * CELL;
    const x2 = px + ch(gx,gy,f*7+4) * CELL;
    const y2 = py + ch(gx,gy,f*7+5) * CELL;
    ctx.fillStyle = shades[f % shades.length];
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2);
    ctx.closePath(); ctx.fill();
  }
  ctx.save(); ctx.beginPath(); ctx.rect(px,py,CELL,CELL); ctx.clip();
  ctx.strokeStyle = 'rgba(15,15,18,0.7)'; ctx.lineWidth = 0.9;
  for (let c = 0; c < 2; c++) {
    const ax = px + ch(gx,gy,c*10+60) * CELL;
    const ay = py + ch(gx,gy,c*10+61) * CELL;
    const bx2= px + ch(gx,gy,c*10+62) * CELL;
    const by2= py + ch(gx,gy,c*10+63) * CELL;
    const mx = (ax+bx2)/2 + (ch(gx,gy,c*10+64)-0.5)*14;
    const my = (ay+by2)/2 + (ch(gx,gy,c*10+65)-0.5)*14;
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.quadraticCurveTo(mx,my,bx2,by2); ctx.stroke();
  }
  const hl = 0.06 + ch(gx,gy,99)*0.07;
  ctx.fillStyle = `rgba(255,255,255,${hl})`;
  ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+CELL*0.75,py); ctx.lineTo(px,py+CELL*0.55);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawTerrain() {
  const { width: W, height: H, terrain, spawn_points } = state;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const t = terrain[y * W + x];
      if (t === 1) {
        drawRock(bx(x), by(y), x, y);
      } else {
        ctx.fillStyle = T_FILL[t];
        ctx.fillRect(bx(x), by(y), CELL, CELL);
      }
      ctx.strokeStyle = 'rgba(0,0,0,0.22)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(bx(x)+.5, by(y)+.5, CELL-1, CELL-1);
      if (t === 2) {
        ctx.strokeStyle = 'rgba(80,160,220,0.2)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 3; i++) {
          const wy = by(y) + 9 + i*10 + Math.sin(pulseT*0.7 + x*0.5 + i) * 1.5;
          ctx.beginPath();
          ctx.moveTo(bx(x)+4, wy);
          ctx.bezierCurveTo(bx(x)+12, wy-3, bx(x)+28, wy+3, bx(x)+CELL-4, wy);
          ctx.stroke();
        }
      }
    }
  }
  const alpha = 0.45 + 0.45 * Math.sin(pulseT * 2);
  for (const [sx, sy] of spawn_points) {
    ctx.strokeStyle = `rgba(240,165,0,${alpha})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([4,3]);
    ctx.strokeRect(bx(sx)+3, by(sy)+3, CELL-6, CELL-6);
    ctx.setLineDash([]);
  }
}

function drawBridge(x, y) {
  const X = bx(x), Y = by(y);
  ctx.fillStyle = '#c8a05a';
  ctx.fillRect(X+2, Y+5, CELL-4, CELL-10);
  ctx.strokeStyle = '#a07838'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const py = Y+5 + (CELL-10)*i/4;
    ctx.beginPath(); ctx.moveTo(X+2,py); ctx.lineTo(X+CELL-2,py); ctx.stroke();
  }
  ctx.strokeStyle = '#8b6020'; ctx.lineWidth = 2;
  ctx.strokeRect(X+2, Y+5, CELL-4, CELL-10);
}

function drawWall(x, y, hp, maxHp) {
  const X = bx(x), Y = by(y);
  const m = 5, wh = CELL*0.44, wy = Y + (CELL-wh)/2;
  const frac = hp / maxHp;
  ctx.fillStyle = frac > 0.5 ? '#8b6914' : '#6b4010';
  ctx.fillRect(X+m, wy, CELL-m*2, wh);
  ctx.strokeStyle = 'rgba(0,0,0,0.28)'; ctx.lineWidth = 1;
  const bw = (CELL-m*2)/3, bh = wh/2;
  for (let row = 0; row < 2; row++) {
    for (let col = -1; col < 4; col++) {
      const ox = row%2===0 ? 0 : bw*0.5;
      ctx.strokeRect(X+m+col*bw+ox, wy+row*bh, bw, bh);
    }
  }
  hpBar(cx(x), wy+wh+4, CELL-8, hp, maxHp, '#e3b341');
}

function drawTower(x, y, hp, maxHp) {
  const CX = cx(x), CY = cy(y);
  const frac = hp / maxHp;
  const bColor = frac > 0.5 ? '#708090' : '#8b6060';
  const mColor = frac > 0.5 ? '#506070' : '#6b3040';
  const bW = CELL*0.54, bH = CELL*0.54;
  const bTop = CY - bH/2;
  const mW = bW*0.22, mH = CELL*0.19;
  ctx.fillStyle = bColor;
  ctx.fillRect(CX-bW/2, bTop, bW, bH);
  ctx.fillStyle = mColor;
  for (let i = 0; i < 3; i++) {
    const mx = CX - bW/2 + (bW/3)*i + bW/6 - mW/2;
    ctx.fillRect(mx, bTop-mH, mW, mH);
  }
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(CX-2, CY-7, 4, 11);
  ctx.strokeStyle = frac > 0.5 ? '#405060' : '#5a2828';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(CX-bW/2, bTop, bW, bH);
  hpBar(CX, CY+bH/2+4, bW+8, hp, maxHp, '#58a6ff');
}

function drawBuilder(bxi, byi, hp, maxHp) {
  const CX = cx(bxi), CY = cy(byi);
  const r = CELL*0.27;
  const g = ctx.createRadialGradient(CX,CY,0,CX,CY,r*2);
  g.addColorStop(0,'rgba(240,192,64,0.28)');
  g.addColorStop(1,'rgba(240,192,64,0)');
  ctx.fillStyle=g; ctx.beginPath(); ctx.arc(CX,CY,r*2,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='rgba(0,0,0,0.25)';
  ctx.beginPath(); ctx.ellipse(CX+1,CY+r+2,r*0.7,r*0.22,0,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#f0c040'; ctx.strokeStyle='#c08000'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(CX,CY+1,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
  ctx.fillStyle='#e05000';
  ctx.beginPath(); ctx.ellipse(CX,CY-r+2,r*0.78,r*0.42,0,Math.PI,0); ctx.fill();
  ctx.fillStyle='#ff6800';
  ctx.fillRect(CX-r*0.95,CY-r+2,r*1.9,3.5);
  hpBar(CX, CY+r+7, CELL-4, hp, maxHp, '#3fb950');
}

function drawEnemies(enemies, maxHp) {
  if (!enemies.length) return;
  const groups = {};
  for (const e of enemies) {
    const k = `${e.x},${e.y}`;
    if (!groups[k]) groups[k] = {x:e.x,y:e.y,count:0,totalHp:0};
    groups[k].count++;
    groups[k].totalHp += e.hp;
  }
  for (const g of Object.values(groups)) {
    const CX = cx(g.x), CY = cy(g.y);
    const r = CELL*0.27;
    const avgHp = g.totalHp / g.count;
    const frac = avgHp / maxHp;
    ctx.fillStyle='rgba(0,0,0,0.28)';
    ctx.beginPath(); ctx.ellipse(CX+1,CY+r+2,r*0.7,r*0.22,0,0,Math.PI*2); ctx.fill();
    ctx.fillStyle = frac>0.5 ? '#c0392b' : '#e74c3c';
    ctx.strokeStyle='#8b1a10'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(CX,CY,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle='#fff';
    ctx.beginPath(); ctx.arc(CX-r*0.29,CY-r*0.15,2.8,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CX+r*0.29,CY-r*0.15,2.8,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#111';
    ctx.beginPath(); ctx.arc(CX-r*0.29+1,CY-r*0.12,1.4,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CX+r*0.29+1,CY-r*0.12,1.4,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle='#5a0000'; ctx.lineWidth=1.8;
    ctx.beginPath(); ctx.moveTo(CX-r*0.52,CY-r*0.38); ctx.lineTo(CX-r*0.08,CY-r*0.26); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX+r*0.08,CY-r*0.26); ctx.lineTo(CX+r*0.52,CY-r*0.38); ctx.stroke();
    if (g.count > 1) {
      ctx.fillStyle='#fff';
      ctx.font=`bold ${Math.round(CELL*0.3)}px sans-serif`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(g.count>9?'9+':String(g.count), CX, CY+r*0.3);
    }
    hpBar(CX, CY+r+5, r*2.2, avgHp, maxHp, '#e74c3c');
  }
}

function drawFrame() {
  if (!state || !state.ready || !state.ticks.length) return;
  const T = state.ticks[tickIdx];
  canvas.width  = state.width  * CELL;
  canvas.height = state.height * CELL;
  drawTerrain();
  for (const s of T.structures.filter(s=>s.kind===3)) drawBridge(s.x,s.y);
  for (const s of T.structures.filter(s=>s.kind===2)) drawWall(s.x,s.y,s.hp,15);
  for (const s of T.structures.filter(s=>s.kind===1)) drawTower(s.x,s.y,s.hp,20);
  drawEnemies(T.enemies, state.enemy_max_hp);
  drawBuilder(T.builder_x, T.builder_y, T.builder_hp, state.builder_max_hp);

  document.getElementById('s-gen').textContent  = state.generation;
  const fitEl = document.getElementById('s-fit');
  fitEl.textContent  = state.fitness.toFixed(1);
  fitEl.className = 'stat-value ' + (state.fitness>=state.max_fitness?'green':state.fitness>200?'gold':'red');
  document.getElementById('s-tick').textContent = `${T.tick}/${state.ticks.length-1}`;
  const bhEl = document.getElementById('s-bhp');
  bhEl.textContent   = T.builder_hp;
  bhEl.className = 'stat-value ' + (T.builder_hp>25?'green':T.builder_hp>0?'gold':'red');
  document.getElementById('s-ene').textContent  = T.enemies.length;
  document.getElementById('s-kill').textContent = T.enemies_killed;

  const activeIdx = T.builder_instr_idx ?? -1;
  const instrs = state.builder_instrs || [];
  if (instrList.childElementCount !== instrs.length) {
    instrList.innerHTML = '';
    instrs.forEach((txt, i) => {
      const div = document.createElement('div');
      div.className = 'instr-item';
      div.innerHTML = `<span class="idx">${i+1}</span><span>${txt}</span>`;
      instrList.appendChild(div);
    });
  }
  instrList.childNodes.forEach((div, i) => {
    if (i < activeIdx) div.className = 'instr-item done';
    else if (i === activeIdx) div.className = 'instr-item active';
    else div.className = 'instr-item';
  });
  if (activeIdx >= 0 && instrList.children[activeIdx]) {
    instrList.children[activeIdx].scrollIntoView({ block: 'nearest' });
  }
  instrPanel.style.maxHeight = canvas.height + 'px';
}

function fetchState() {
  if (fetching) return;
  fetching = true;
  fetch('/tower-defence/api/state')
    .then(r => r.json())
    .then(data => {
      if (data.ready && data.ticks && data.ticks.length > 0) {
        state = data;
        tickIdx = 0;
        instrList.innerHTML = '';
        loading.style.display = 'none';
      }
    })
    .catch(()=>{})
    .finally(()=>{ fetching = false; });
}

const TICK_MS = 150;
let lastMs = 0;

function loop(now) {
  pulseT = now * 0.001;
  if (state && state.ready) {
    if (now - lastMs >= TICK_MS) {
      lastMs = now;
      tickIdx++;
      if (tickIdx >= state.ticks.length) {
        tickIdx = state.ticks.length - 1;
        fetchState();
      }
    }
    drawFrame();
  } else {
    fetchState();
  }
  requestAnimationFrame(loop);
}

fetchState();
requestAnimationFrame(loop);
</script>
</body>
</html>
"####;
}

// ─── Root index ───────────────────────────────────────────────────────────────

const ROOT_HTML: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GA Examples</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:'Courier New',monospace;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:100vh;gap:32px;padding:40px 16px;
}
h1{color:#58a6ff;font-size:1.4rem;letter-spacing:4px;text-transform:uppercase}
p{color:#8b949e;font-size:12px;max-width:480px;text-align:center;line-height:1.6}
.cards{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
.card{
  background:#161b22;border:1px solid #30363d;border-radius:12px;
  padding:24px 28px;min-width:200px;max-width:240px;
  display:flex;flex-direction:column;gap:10px;
  text-decoration:none;color:inherit;
  transition:border-color 0.15s,transform 0.1s;
}
.card:hover{border-color:#58a6ff;transform:translateY(-2px)}
.card-title{color:#58a6ff;font-size:1rem;letter-spacing:2px;text-transform:uppercase}
.card-desc{color:#8b949e;font-size:11px;line-height:1.5}
</style>
</head>
<body>
<h1>Genetic Algorithm Examples</h1>
<p>Three live simulations — each running a GA in the background. Click to explore.</p>
<div class="cards">
  <a href="/swarm" class="card">
    <div class="card-title">Swarm</div>
    <div class="card-desc">Evolves 4 steering weights for a boid swarm. GA runs every tick, best weights applied immediately.</div>
  </a>
  <a href="/maze" class="card">
    <div class="card-title">Maze</div>
    <div class="card-desc">Evolves a sequence of turn decisions to navigate a procedurally generated maze.</div>
  </a>
  <a href="/tower-defence" class="card">
    <div class="card-title">Tower Defence</div>
    <div class="card-desc">Evolves a builder's instruction queue — towers, walls, bridges — to survive 300 ticks.</div>
  </a>
</div>
</body>
</html>
"####;

async fn handle_root() -> Html<&'static str> { Html(ROOT_HTML) }

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
