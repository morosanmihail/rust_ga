#[allow(dead_code)]
mod swarm;

use std::sync::{Arc, RwLock};

use axum::{extract::State, response::Html, routing::get, Json, Router};
use ga::population::{MutationConfig, Population, PopulationConfig};
use rand::Rng;
use serde::Serialize;

use swarm::evolve::{update_snapshot, SteeringGenome};
use swarm::sim::{SwarmSim, WORLD_H, WORLD_W};

const GA_GENS_PER_TICK: usize = 20;
const MAX_SIM_TICKS: u64 = 300;
const TICK_SLEEP_MS: u64 = 50;

// ── API types ─────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct AgentSnap {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    done: bool,
}

#[derive(Serialize, Clone, Default)]
struct ApiState {
    tick: u64,
    ga_gens: u64,
    run: u64,
    agents: Vec<AgentSnap>,
    obstacles: Vec<[f64; 3]>,
    /// [x, y, w, h] of the goal rectangle.
    goal: [f64; 4],
    world_w: f64,
    world_h: f64,
    weights: [f32; 4],
    agents_at_goal: usize,
    total_agents: usize,
    finished: bool,
}

// ── GA background thread ──────────────────────────────────────────────────────

fn make_pop() -> Population<SteeringGenome> {
    Population::new(PopulationConfig {
        seed: rand::thread_rng().gen(),
        pop_size: 40,
        elitism_count: 4,
        mutate_count: 16,
        crossover_count: 10,
        mutation_config: MutationConfig { gene_mutation_chance: 0.5 },
        preseeded_population: vec![],
    })
}

fn ga_thread(shared: Arc<RwLock<ApiState>>) {
    let mut run = 0u64;
    let mut total_ga_gens = 0u64;

    loop {
        run += 1;
        let mut sim = SwarmSim::new_random(rand::thread_rng().gen());
        update_snapshot(&sim);

        let obstacles: Vec<[f64; 3]> = sim.obstacles.iter()
            .map(|o| [o.pos.0, o.pos.1, o.radius])
            .collect();

        let mut pop = make_pop();

        loop {
            for _ in 0..GA_GENS_PER_TICK {
                pop.tick_parallel();
                total_ga_gens += 1;
            }

            let best = pop.get_best_member().clone();
            sim.step(&best.to_weights());
            update_snapshot(&sim);

            let agents: Vec<AgentSnap> = sim.agents.iter()
                .map(|a| AgentSnap {
                    x: a.pos.0, y: a.pos.1,
                    vx: a.vel.0, vy: a.vel.1,
                    done: a.reached_goal,
                })
                .collect();

            let finished = sim.agents_at_goal() == sim.agents.len()
                || sim.tick >= MAX_SIM_TICKS;

            *shared.write().unwrap() = ApiState {
                tick: sim.tick,
                ga_gens: total_ga_gens,
                run,
                agents,
                obstacles: obstacles.clone(),
                goal: [sim.goal.x, sim.goal.y, sim.goal.w, sim.goal.h],
                world_w: WORLD_W,
                world_h: WORLD_H,
                weights: best.weights,
                agents_at_goal: sim.agents_at_goal(),
                total_agents: sim.agents.len(),
                finished,
            };

            if finished {
                std::thread::sleep(std::time::Duration::from_secs(4));
                break;
            }

            std::thread::sleep(std::time::Duration::from_millis(TICK_SLEEP_MS));
        }
    }
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

async fn handle_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn handle_state(State(s): State<Arc<RwLock<ApiState>>>) -> Json<ApiState> {
    Json(s.read().unwrap().clone())
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let shared = Arc::new(RwLock::new(ApiState::default()));
    let shared_ga = shared.clone();

    std::thread::spawn(move || ga_thread(shared_ga));

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/api/state", get(handle_state))
        .with_state(shared);

    println!("Swarm GA server on http://localhost:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Embedded frontend ─────────────────────────────────────────────────────────

const INDEX_HTML: &str = r####"<!DOCTYPE html>
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
#footer{font-size:10px;color:#484f58;letter-spacing:1px}
#loading{font-size:14px;color:#8b949e;padding:40px}
</style>
</head>
<body>
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

  // Background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, CW, CH);

  // Grid lines (faint)
  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= state.world_w; x++) {
    ctx.beginPath(); ctx.moveTo(x*SCALE,0); ctx.lineTo(x*SCALE,CH); ctx.stroke();
  }
  for (let y = 0; y <= state.world_h; y++) {
    ctx.beginPath(); ctx.moveTo(0,y*SCALE); ctx.lineTo(CW,y*SCALE); ctx.stroke();
  }

  // Obstacles
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

  // Goal rectangle
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

  // Agent velocity arrows + bodies
  for (const a of state.agents) {
    const ax = a.x * SCALE, ay = a.y * SCALE;

    if (!a.done) {
      // Velocity arrow
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

    // Body
    const r = a.done ? 5 : 4;
    ctx.fillStyle   = a.done ? '#3fb950' : '#58a6ff';
    ctx.strokeStyle = a.done ? '#2ea043' : '#1f6feb';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(ax, ay, r, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  }

  // Finished overlay
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

  // Stat text
  document.getElementById('s-run').textContent  = state.run;
  document.getElementById('s-tick').textContent = state.tick;
  document.getElementById('s-gens').textContent = state.ga_gens.toLocaleString();
  const goalEl = document.getElementById('s-goal');
  goalEl.textContent = `${state.agents_at_goal}/${state.total_agents}`;
  goalEl.className = 'stat-value ' + (state.agents_at_goal === state.total_agents ? 'green' : '');

  // Weight bars
  for (let i = 0; i < 4; i++) {
    const v = state.weights[i];
    const pct = Math.min(1, v / MAX_W);
    document.getElementById(`w${i}`).style.height = (pct * 50) + 'px';
    document.getElementById(`wv${i}`).textContent = v.toFixed(2);
  }
}

function poll() {
  fetch('/api/state')
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
