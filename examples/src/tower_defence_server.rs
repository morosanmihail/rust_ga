#[allow(dead_code)]
mod tower_defence;

use std::sync::{Arc, RwLock};

use axum::{extract::State, response::Html, routing::get, Json, Router};
use ga::population::{MutationConfig, Population, PopulationConfig};
use ga::traits::FitnessRetrieve;
use rand::Rng;
use serde::Serialize;

use tower_defence::{
    builder::{Builder, Instruction},
    evolve::{eval_config, make_eval_map, max_fitness, set_map_seed, BuilderGenome, BUILDER_HP, BUILDER_START},
    map::{Structure, Terrain},
    simulation::Simulation,
};

// ── API snapshot types ────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct EnemySnap {
    x: usize,
    y: usize,
    hp: u32,
}

#[derive(Serialize, Clone)]
struct StructSnap {
    x: usize,
    y: usize,
    kind: u8,  // 1=Tower  2=Wall  3=Bridge
    hp: u32,   // 0 for bridge
}

#[derive(Serialize, Clone)]
struct TickSnap {
    tick: u64,
    builder_x: usize,
    builder_y: usize,
    builder_hp: u32,
    builder_instr_idx: Option<usize>,
    enemies_killed: u32,
    enemies: Vec<EnemySnap>,
    structures: Vec<StructSnap>,
}

#[derive(Serialize, Clone, Default)]
struct ApiState {
    width: usize,
    height: usize,
    generation: usize,
    fitness: f64,
    max_fitness: f64,
    terrain: Vec<u8>,              // 0=plain  1=rock  2=water  (row-major)
    spawn_points: Vec<[usize; 2]>,
    builder_max_hp: u32,
    enemy_max_hp: u32,
    builder_instrs: Vec<String>,   // human-readable instruction list (fixed per genome)
    ticks: Vec<TickSnap>,
    survived: bool,
    ready: bool,
}

// ── Simulation recorder ───────────────────────────────────────────────────────

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
        builder_x: sim.builder.x,
        builder_y: sim.builder.y,
        builder_hp: sim.builder.hp,
        builder_instr_idx: sim.builder.current_instr_idx,
        enemies_killed: sim.enemies_killed,
        enemies,
        structures,
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

// ── GA background thread ──────────────────────────────────────────────────────

fn make_pop_config() -> PopulationConfig<BuilderGenome> {
    PopulationConfig {
        pop_size: 80,
        crossover_count: 25,
        mutate_count: 25,
        elitism_count: 5,
        mutation_config: MutationConfig { gene_mutation_chance: 0.25 },
        seed: rand::thread_rng().gen(),
        preseeded_population: vec![],
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

fn ga_thread(shared: Arc<RwLock<ApiState>>) {
    let mut rng = rand::thread_rng();
    let cfg = eval_config();

    set_map_seed(rng.gen());
    let (mut width, mut height, mut terrain, mut spawn_points) = make_map_info();
    let mut mf = max_fitness();

    let mut pop: Population<BuilderGenome> = Population::new(make_pop_config());
    let mut gen = 0usize;

    loop {
        std::thread::sleep(std::time::Duration::from_millis(2500));

        pop.tick_parallel();
        gen += 1;

        let best = pop.get_best_member().clone();
        let fitness = best.get_fitness().unwrap_or(0.0);
        let (builder_instrs, ticks, survived) = record_sim(&best);

        *shared.write().unwrap() = ApiState {
            width,
            height,
            generation: gen,
            fitness,
            max_fitness: mf,
            terrain: terrain.clone(),
            spawn_points: spawn_points.clone(),
            builder_max_hp: BUILDER_HP,
            enemy_max_hp: cfg.enemy_hp,
            builder_instrs,
            ticks,
            survived,
            ready: true,
        };

        if fitness >= mf || gen >= 150 {
            let solved = fitness >= mf;
            let reason = if solved { "solved" } else { "gen limit" };
            println!("Gen {gen}: {reason} (fitness {fitness:.1}), resetting");
            if solved {
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
            set_map_seed(rng.gen());
            (width, height, terrain, spawn_points) = make_map_info();
            mf = max_fitness();
            pop = Population::new(make_pop_config());
            gen = 0;
        } else {
            println!("Gen {gen}: fitness {fitness:.1}");
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

    let addr = "0.0.0.0:3000";
    println!("Tower Defence GA server running on http://localhost:3000");
    println!("First generation in 5 seconds…");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Embedded frontend ─────────────────────────────────────────────────────────

const INDEX_HTML: &str = r####"<!DOCTYPE html>
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
#footer{font-size:10px;color:#484f58;letter-spacing:1px;margin-top:2px}
#loading{font-size:14px;color:#8b949e;padding:40px}
</style>
</head>
<body>
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

// ── Helpers ──────────────────────────────────────────────────────────────────
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

// ── Terrain & spawns ─────────────────────────────────────────────────────────
const T_FILL   = ['#2d5a3d','#4a4a50','#0a3d62'];

// Deterministic per-cell hash — returns [0,1) for seed n at grid cell (gx,gy).
function ch(gx, gy, n) {
  let v = Math.imul(gx * 1664525 + n * 6364136, gy * 1013904223 + n * 22695477);
  v ^= v >>> 13; v = Math.imul(v ^ 0x9e3779b9, 0x6c62272e);
  v ^= v >>> 15;
  return ((v >>> 0) & 0xffff) / 0x10000;
}

function drawRock(px, py, gx, gy) {
  // base
  ctx.fillStyle = '#3a3a40';
  ctx.fillRect(px, py, CELL, CELL);

  // 5 triangular facets — slight tone variation gives stone feel
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

  // clip subsequent drawing to cell
  ctx.save(); ctx.beginPath(); ctx.rect(px,py,CELL,CELL); ctx.clip();

  // 2 jagged cracks
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

  // top-left highlight
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
      // subtle grid
      ctx.strokeStyle = 'rgba(0,0,0,0.22)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(bx(x)+.5, by(y)+.5, CELL-1, CELL-1);
      // water ripples
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
  // spawn points — pulsing dashed rectangle
  const alpha = 0.45 + 0.45 * Math.sin(pulseT * 2);
  for (const [sx, sy] of spawn_points) {
    ctx.strokeStyle = `rgba(240,165,0,${alpha})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([4,3]);
    ctx.strokeRect(bx(sx)+3, by(sy)+3, CELL-6, CELL-6);
    ctx.setLineDash([]);
  }
}

// ── Bridge ───────────────────────────────────────────────────────────────────
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

// ── Wall ─────────────────────────────────────────────────────────────────────
function drawWall(x, y, hp, maxHp) {
  const X = bx(x), Y = by(y);
  const m = 5, wh = CELL*0.44, wy = Y + (CELL-wh)/2;
  const frac = hp / maxHp;
  ctx.fillStyle = frac > 0.5 ? '#8b6914' : '#6b4010';
  ctx.fillRect(X+m, wy, CELL-m*2, wh);
  // brick lines
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

// ── Tower ────────────────────────────────────────────────────────────────────
function drawTower(x, y, hp, maxHp) {
  const CX = cx(x), CY = cy(y);
  const frac = hp / maxHp;
  const bColor = frac > 0.5 ? '#708090' : '#8b6060';
  const mColor = frac > 0.5 ? '#506070' : '#6b3040';
  const bW = CELL*0.54, bH = CELL*0.54;
  const bTop = CY - bH/2;
  const mW = bW*0.22, mH = CELL*0.19;

  // body
  ctx.fillStyle = bColor;
  ctx.fillRect(CX-bW/2, bTop, bW, bH);
  // 3 merlons
  ctx.fillStyle = mColor;
  for (let i = 0; i < 3; i++) {
    const mx = CX - bW/2 + (bW/3)*i + bW/6 - mW/2;
    ctx.fillRect(mx, bTop-mH, mW, mH);
  }
  // arrow slit
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(CX-2, CY-7, 4, 11);
  // outline
  ctx.strokeStyle = frac > 0.5 ? '#405060' : '#5a2828';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(CX-bW/2, bTop, bW, bH);
  hpBar(CX, CY+bH/2+4, bW+8, hp, maxHp, '#58a6ff');
}

// ── Builder ──────────────────────────────────────────────────────────────────
function drawBuilder(bxi, byi, hp, maxHp) {
  const CX = cx(bxi), CY = cy(byi);
  const r = CELL*0.27;
  // glow
  const g = ctx.createRadialGradient(CX,CY,0,CX,CY,r*2);
  g.addColorStop(0,'rgba(240,192,64,0.28)');
  g.addColorStop(1,'rgba(240,192,64,0)');
  ctx.fillStyle=g; ctx.beginPath(); ctx.arc(CX,CY,r*2,0,Math.PI*2); ctx.fill();
  // shadow
  ctx.fillStyle='rgba(0,0,0,0.25)';
  ctx.beginPath(); ctx.ellipse(CX+1,CY+r+2,r*0.7,r*0.22,0,0,Math.PI*2); ctx.fill();
  // body
  ctx.fillStyle='#f0c040'; ctx.strokeStyle='#c08000'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(CX,CY+1,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
  // hard hat dome
  ctx.fillStyle='#e05000';
  ctx.beginPath(); ctx.ellipse(CX,CY-r+2,r*0.78,r*0.42,0,Math.PI,0); ctx.fill();
  // brim
  ctx.fillStyle='#ff6800';
  ctx.fillRect(CX-r*0.95,CY-r+2,r*1.9,3.5);
  hpBar(CX, CY+r+7, CELL-4, hp, maxHp, '#3fb950');
}

// ── Enemies ──────────────────────────────────────────────────────────────────
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
    // shadow
    ctx.fillStyle='rgba(0,0,0,0.28)';
    ctx.beginPath(); ctx.ellipse(CX+1,CY+r+2,r*0.7,r*0.22,0,0,Math.PI*2); ctx.fill();
    // body
    ctx.fillStyle = frac>0.5 ? '#c0392b' : '#e74c3c';
    ctx.strokeStyle='#8b1a10'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(CX,CY,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
    // eyes
    ctx.fillStyle='#fff';
    ctx.beginPath(); ctx.arc(CX-r*0.29,CY-r*0.15,2.8,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CX+r*0.29,CY-r*0.15,2.8,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#111';
    ctx.beginPath(); ctx.arc(CX-r*0.29+1,CY-r*0.12,1.4,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CX+r*0.29+1,CY-r*0.12,1.4,0,Math.PI*2); ctx.fill();
    // angry brows
    ctx.strokeStyle='#5a0000'; ctx.lineWidth=1.8;
    ctx.beginPath(); ctx.moveTo(CX-r*0.52,CY-r*0.38); ctx.lineTo(CX-r*0.08,CY-r*0.26); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX+r*0.08,CY-r*0.26); ctx.lineTo(CX+r*0.52,CY-r*0.38); ctx.stroke();
    // count overlay
    if (g.count > 1) {
      ctx.fillStyle='#fff';
      ctx.font=`bold ${Math.round(CELL*0.3)}px sans-serif`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(g.count>9?'9+':String(g.count), CX, CY+r*0.3);
    }
    hpBar(CX, CY+r+5, r*2.2, avgHp, maxHp, '#e74c3c');
  }
}

// ── Full frame ────────────────────────────────────────────────────────────────
function drawFrame() {
  if (!state || !state.ready || !state.ticks.length) return;
  const T = state.ticks[tickIdx];
  canvas.width  = state.width  * CELL;
  canvas.height = state.height * CELL;

  drawTerrain();

  // structures: bridges → walls → towers
  for (const s of T.structures.filter(s=>s.kind===3)) drawBridge(s.x,s.y);
  for (const s of T.structures.filter(s=>s.kind===2)) drawWall(s.x,s.y,s.hp,state.enemy_max_hp>0?15:15);
  for (const s of T.structures.filter(s=>s.kind===1)) drawTower(s.x,s.y,s.hp,20);

  drawEnemies(T.enemies, state.enemy_max_hp);
  drawBuilder(T.builder_x, T.builder_y, T.builder_hp, state.builder_max_hp);

  // stats
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

  // Instruction panel
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

// ── Fetch state ───────────────────────────────────────────────────────────────
function fetchState() {
  if (fetching) return;
  fetching = true;
  fetch('/api/state')
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

// ── Animation loop ────────────────────────────────────────────────────────────
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
