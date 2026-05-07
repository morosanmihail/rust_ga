use std::{collections::VecDeque, sync::{Arc, OnceLock}};

use ga::{
    population::{MutationConfig, Population, PopulationConfig},
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use image::{ImageBuffer, Rgb};
use rand::{rngs::StdRng, Rng, SeedableRng};

const IMAGE_SIZE: u32 = 128;
const MAX_DEPTH: usize = 4;
const GENERATIONS: usize = 3000;
const OUTPUT_DIR: &str = "image_output";

// --- Expression tree ---

type Child = Option<Arc<Node>>;

#[derive(Default, Clone, Debug)]
struct Tree {
    root: Child,
}

#[derive(Default, Clone, Debug)]
struct Node {
    value: String,
    left: Child,
    right: Child,
}

impl Node {
    fn new(value: String, left: Child, right: Child) -> Child {
        Some(Arc::new(Node { value, left, right }))
    }



    fn node_count(&self) -> usize {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => 1 + l.node_count() + r.node_count(),
            _ => 1,
        }
    }

    fn depth(&self) -> usize {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => 1 + l.depth().max(r.depth()),
            _ => 0,
        }
    }

    fn print(&self) -> String {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => format!("({} {} {})", l.print(), self.value, r.print()),
            _ => self.value.clone(),
        }
    }
}

fn get_nth_node(root: &Child, n: usize) -> Child {
    let mut queue: VecDeque<Child> = VecDeque::new();
    queue.push_back(root.clone());
    let mut idx = 0;
    while let Some(maybe_node) = queue.pop_front() {
        if let Some(node) = &maybe_node {
            if idx == n {
                return maybe_node.clone();
            }
            idx += 1;
            queue.push_back(node.left.clone());
            queue.push_back(node.right.clone());
        }
    }
    root.clone()
}

fn random_node(depth: usize, seed: [u8; 32]) -> Child {
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    if depth == 0 {
        return match rng.gen_range(0u8..3) {
            0 => Node::new("x".into(), None, None),
            1 => Node::new("y".into(), None, None),
            _ => Node::new(rng.gen_range(-16i64..16).to_string(), None, None),
        };
    }
    let ops = ["+", "-", "*", "/", "s", "c", "d"];
    let op = ops[rng.gen_range(0..ops.len())].to_string();
    Node::new(op, random_node(depth - 1, rng.gen()), random_node(depth - 1, rng.gen()))
}

fn mutate_node(root: &Child, config: &MutationConfig, rng: &mut StdRng) -> Child {
    match root {
        None => None,
        Some(node) => {
            if rng.gen::<f64>() < config.gene_mutation_chance {
                let is_leaf = node.left.is_none();
                if is_leaf {
                    if let Ok(val) = node.value.parse::<i64>() {
                        // Nudge constant by a small delta instead of replacing
                        let delta = rng.gen_range(-1i64..=1);
                        Node::new((val + delta).clamp(-255, 255).to_string(), None, None)
                    } else {
                        // Variable leaf: swap to another leaf
                        random_node(0, rng.gen())
                    }
                } else if rng.gen::<f64>() < 0.03 {
                    // Rare structural mutation: replace whole subtree
                    random_node(2, rng.gen())
                } else {
                    // Common: swap operator only, keep children intact
                    let ops = ["+", "-", "*", "/", "s", "c", "d"];
                    let op = ops[rng.gen_range(0..ops.len())].to_string();
                    Node::new(op, node.left.clone(), node.right.clone())
                }
            } else {
                let new_left = mutate_node(&node.left, config, rng);
                let new_right = mutate_node(&node.right, config, rng);
                Node::new(node.value.clone(), new_left, new_right)
            }
        }
    }
}

fn replace_nth(root: &Child, target: usize, donor: &Child, idx: &mut usize) -> Child {
    match root {
        None => None,
        Some(node) => {
            let my_idx = *idx;
            *idx += 1;
            if my_idx == target {
                donor.clone()
            } else {
                let new_left = replace_nth(&node.left, target, donor, idx);
                let new_right = replace_nth(&node.right, target, donor, idx);
                Node::new(node.value.clone(), new_left, new_right)
            }
        }
    }
}

fn crossover_trees(a: &Tree, b: &Tree, rng: &mut StdRng) -> Tree {
    let a_count = a.root.as_ref().map(|n| n.node_count()).unwrap_or(1);
    let b_count = b.root.as_ref().map(|n| n.node_count()).unwrap_or(1);
    let donor = get_nth_node(&b.root, rng.gen_range(0..b_count));
    let mut idx = 0;
    Tree {
        root: replace_nth(&a.root, rng.gen_range(0..a_count), &donor, &mut idx),
    }
}

// --- Bytecode compiler + stack-machine evaluator ---
// Replaces recursive Arc<Node> traversal: no pointer chasing, no per-pixel heap alloc.

#[derive(Clone, Copy, Debug)]
enum Op {
    PushX,
    PushY,
    PushConst(f64),
    Add, Sub, Mul, Div, Sin, Cos, Dist,
}

fn compile_tree(node: &Child, ops: &mut Vec<Op>) {
    match node {
        None => ops.push(Op::PushConst(128.0)),
        Some(n) if n.left.is_none() => match n.value.as_str() {
            "x" => ops.push(Op::PushX),
            "y" => ops.push(Op::PushY),
            _ => ops.push(Op::PushConst(n.value.parse().unwrap_or(0.0))),
        },
        Some(n) => {
            compile_tree(&n.left, ops);
            compile_tree(&n.right, ops);
            ops.push(match n.value.as_str() {
                "-" => Op::Sub,
                "*" => Op::Mul,
                "/" => Op::Div,
                "s" => Op::Sin,
                "c" => Op::Cos,
                "d" => Op::Dist,
                _   => Op::Add,
            });
        }
    }
}

fn eval_ops(ops: &[Op], x: f64, y: f64) -> f64 {
    // Fixed stack: depth-10 balanced tree needs at most depth+1=11 slots.
    let mut stack = [0.0f64; 64];
    let mut top = 0usize;
    for op in ops {
        match op {
            Op::PushX        => { stack[top] = x;      top += 1; }
            Op::PushY        => { stack[top] = y;      top += 1; }
            Op::PushConst(v) => { stack[top] = *v;     top += 1; }
            Op::Add  => { top -= 1; stack[top-1] += stack[top]; }
            Op::Sub  => { top -= 1; stack[top-1] -= stack[top]; }
            Op::Mul  => { top -= 1; stack[top-1] *= stack[top]; }
            Op::Div  => { top -= 1; stack[top-1] = if stack[top].abs() < 1e-10 { 0.0 } else { stack[top-1] / stack[top] }; }
            Op::Sin  => { top -= 1; stack[top-1] = stack[top-1].sin() * stack[top]; }
            Op::Cos  => { top -= 1; stack[top-1] = stack[top-1].cos() * stack[top]; }
            Op::Dist => { top -= 1; stack[top-1] = (stack[top-1]*stack[top-1] + stack[top]*stack[top]).sqrt(); }
        }
    }
    if top == 0 { return 128.0; }
    let v = stack[top - 1];
    if v.is_nan() || v.is_infinite() { 128.0 } else { v.clamp(0.0, 255.0) }
}

// --- Target image (plasma pattern, no external file needed) ---

fn target_pixel(x: u32, y: u32) -> [f64; 3] {
    let fx = x as f64;
    let fy = y as f64;

    // Each channel is depth-4 using our operators: *(d(s(/(x,8),y), c(/(y,8),x)), 3)
    let r = ((fx / 8.0).sin() * fy).hypot((fy / 8.0).cos() * fx) * 3.0;
    let g = ((fy / 8.0).sin() * fx).hypot((fx / 8.0).cos() * fy) * 3.0;
    let b = ((fx / 8.0).sin() * fx).hypot((fy / 8.0).cos() * fy) * 3.0;

    [r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), b.clamp(0.0, 255.0)]
}

static TARGET: OnceLock<Vec<[f64; 3]>> = OnceLock::new();
static TARGET_VARIANCE: OnceLock<[f64; 3]> = OnceLock::new();

fn get_target() -> &'static Vec<[f64; 3]> {
    TARGET.get_or_init(|| {
        (0..IMAGE_SIZE)
            .flat_map(|y| (0..IMAGE_SIZE).map(move |x| target_pixel(x, y)))
            .collect()
    })
}

fn get_target_variance() -> &'static [f64; 3] {
    TARGET_VARIANCE.get_or_init(|| {
        let pixels = get_target();
        let n = pixels.len() as f64;
        let mut sum = [0.0f64; 3];
        let mut sum2 = [0.0f64; 3];
        for px in pixels {
            for ch in 0..3 {
                sum[ch] += px[ch];
                sum2[ch] += px[ch] * px[ch];
            }
        }
        std::array::from_fn(|ch| (sum2[ch] / n) - (sum[ch] / n).powi(2))
    })
}

// --- GA genome ---

#[derive(Default, Clone, Debug)]
struct RgbTree {
    r: Tree,
    g: Tree,
    b: Tree,
    fitness: Option<f64>,
}

impl Generate for RgbTree {
    fn generate(seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        RgbTree {
            r: Tree { root: random_node(MAX_DEPTH, rng.gen()) },
            g: Tree { root: random_node(MAX_DEPTH, rng.gen()) },
            b: Tree { root: random_node(MAX_DEPTH, rng.gen()) },
            fitness: None,
        }
    }
}

impl Mutate for RgbTree {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        RgbTree {
            r: Tree { root: mutate_node(&self.r.root, config, &mut rng) },
            g: Tree { root: mutate_node(&self.g.root, config, &mut rng) },
            b: Tree { root: mutate_node(&self.b.root, config, &mut rng) },
            fitness: None,
        }
    }
}

impl Crossover for RgbTree {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        RgbTree {
            r: crossover_trees(&self.r, &other.r, &mut rng),
            g: crossover_trees(&self.g, &other.g, &mut rng),
            b: crossover_trees(&self.b, &other.b, &mut rng),
            fitness: None,
        }
    }
}

impl FitnessRetrieve for RgbTree {
    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}

impl Fitness for RgbTree {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        if self.fitness.is_some() {
            return self.fitness;
        }
        let target = get_target();
        let n = (IMAGE_SIZE * IMAGE_SIZE) as f64;
        let mut total_sq_err = 0.0f64;
        let (mut rs, mut gs, mut bs) = (0.0f64, 0.0f64, 0.0f64);
        let (mut rs2, mut gs2, mut bs2) = (0.0f64, 0.0f64, 0.0f64);

        // Compile each tree once, then evaluate over all pixels with no pointer chasing.
        let mut ops_r = Vec::new(); compile_tree(&self.r.root, &mut ops_r);
        let mut ops_g = Vec::new(); compile_tree(&self.g.root, &mut ops_g);
        let mut ops_b = Vec::new(); compile_tree(&self.b.root, &mut ops_b);

        for (i, px) in target.iter().enumerate() {
            let x = (i % IMAGE_SIZE as usize) as f64;
            let y = (i / IMAGE_SIZE as usize) as f64;
            let r = eval_ops(&ops_r, x, y);
            let g = eval_ops(&ops_g, x, y);
            let b = eval_ops(&ops_b, x, y);
            total_sq_err += (r - px[0]).powi(2)
                + (g - px[1]).powi(2)
                + (b - px[2]).powi(2);
            rs += r; rs2 += r * r;
            gs += g; gs2 += g * g;
            bs += b; bs2 += b * b;
        }

        let base = -total_sq_err / n;

        // Variance per channel vs target variance. Score 0..1: how well pred variety matches target.
        let pred_var = [
            (rs2 / n) - (rs / n).powi(2),
            (gs2 / n) - (gs / n).powi(2),
            (bs2 / n) - (bs / n).powi(2),
        ];
        let tgt_var = get_target_variance();
        let mean_variety = pred_var.iter().zip(tgt_var.iter())
            .map(|(p, t)| if *t < 1.0 { 1.0 } else { (p / t).min(1.0) })
            .sum::<f64>() / 3.0;
        let variety_reward = (mean_variety * base.abs()).min(base.abs() * 0.2);

        let total_depth = [&self.r, &self.g, &self.b]
            .iter()
            .map(|t| t.root.as_ref().map(|n| n.depth()).unwrap_or(0))
            .sum::<usize>() as f64;
        // Penalty proportional to depth, capped at 10% of the base score
        let depth_penalty = (total_depth * 50.0).min(base.abs() * 0.1);

        self.fitness = Some(base + variety_reward - depth_penalty);
        self.fitness
    }
}

// --- Rendering ---

fn save_genome_image(genome: &RgbTree, path: &str) {
    let mut ops_r = Vec::new(); compile_tree(&genome.r.root, &mut ops_r);
    let mut ops_g = Vec::new(); compile_tree(&genome.g.root, &mut ops_g);
    let mut ops_b = Vec::new(); compile_tree(&genome.b.root, &mut ops_b);
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(IMAGE_SIZE, IMAGE_SIZE, |x, y| {
            Rgb([
                eval_ops(&ops_r, x as f64, y as f64) as u8,
                eval_ops(&ops_g, x as f64, y as f64) as u8,
                eval_ops(&ops_b, x as f64, y as f64) as u8,
            ])
        });
    img.save(path).expect("Failed to save image");
}

fn save_target_image(path: &str) {
    let target = get_target();
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(IMAGE_SIZE, IMAGE_SIZE, |x, y| {
            let [r, g, b] = target[(y * IMAGE_SIZE + x) as usize];
            Rgb([r as u8, g as u8, b as u8])
        });
    img.save(path).expect("Failed to save target image");
}


fn main() {
    std::fs::create_dir_all(OUTPUT_DIR).unwrap();
    save_target_image(&format!("{OUTPUT_DIR}/target.png"));
    println!("Target saved to {OUTPUT_DIR}/target.png");

    let config = PopulationConfig {
        pop_size: 50,
        crossover_count: 15,
        mutate_count: 15,
        elitism_count: 3,
        mutation_config: MutationConfig {
            gene_mutation_chance: 0.15,
        },
        seed: [41; 32],
        preseeded_population: vec![],
    };

    let mut p: Population<RgbTree> = Population::new(config);
    let mut last_saved_fitness = f64::NEG_INFINITY;
    let mut frame = 0usize;

    for gen in 0..GENERATIONS {
        p.tick_parallel();
        let best = p.get_best_member().clone();
        let fitness = best.get_fitness().unwrap_or(f64::NEG_INFINITY);

        if fitness > last_saved_fitness {
            last_saved_fitness = fitness;
            let path = format!("{OUTPUT_DIR}/frame_{frame:05}.png");
            save_genome_image(&best, &path);
            frame += 1;
            println!(
                "Gen {gen:5}: fitness = {fitness:.1}  R={} G={} B={}  -> {path}",
                best.r.root.as_ref().map(|n| n.print()).unwrap_or_default(),
                best.g.root.as_ref().map(|n| n.print()).unwrap_or_default(),
                best.b.root.as_ref().map(|n| n.print()).unwrap_or_default(),
            );
        } else {
            println!("Gen {gen:5}: fitness = {fitness:.1}");
        }
    }

    println!("Done. Frames in {OUTPUT_DIR}/. Combine with:");
    println!("  ffmpeg -framerate 10 -pattern_type glob -i '{OUTPUT_DIR}/gen_*.png' -vf scale=512:512 evolution.gif");
}
