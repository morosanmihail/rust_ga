use std::{collections::VecDeque, rc::Rc};

use ga::{
    population::{Genome, MutationConfig, Population, PopulationConfig},
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub const MAX_DEPTH: usize = 5;
pub const NR_VARS: usize = 2;
pub const MAX_VALUE: i64 = 255;
pub const MIN_VALUE: i64 = -255;

type Child = Option<Rc<Node>>;

#[derive(Default, Clone, Debug)]
pub struct Tree {
    pub root: Child,
}

#[derive(Default, Clone, Debug)]
pub struct Node {
    value: String,
    left: Child,
    right: Child,
}

impl Node {
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        match (self.value.get(0..1).unwrap(), self.value.len()) {
            ("+", 1) => {
                self.left.as_ref().unwrap().evaluate(x1, x2)
                    + self.right.as_ref().unwrap().evaluate(x1, x2)
            }
            ("-", 1) => {
                self.left.as_ref().unwrap().evaluate(x1, x2)
                    - self.right.as_ref().unwrap().evaluate(x1, x2)
            }
            ("*", 1) => {
                self.left.as_ref().unwrap().evaluate(x1, x2)
                    * self.right.as_ref().unwrap().evaluate(x1, x2)
            }
            ("/", 1) => {
                let r_val = self.right.as_ref().unwrap().evaluate(x1, x2);
                if r_val == 0.0 {
                    0.0
                } else {
                    self.left.as_ref().unwrap().evaluate(x1, x2) / r_val
                }
            }
            ("^", 1) => self
                .left
                .as_ref()
                .unwrap()
                .evaluate(x1, x2)
                .powf(self.right.as_ref().unwrap().evaluate(x1, x2)),
            ("X", _) => match self.value.get(1..2).unwrap() {
                "0" => x1,
                "1" => x2,
                _ => 0.0,
            },
            _ => match self.value.parse() {
                Ok(v) => v,
                _ => 0.0,
            },
        }
    }

    pub fn print(&self) -> String {
        match (self.left.as_ref(), self.right.as_ref()) {
            (Some(l), Some(r)) => {
                let l_res = l.print();
                let r_res = r.print();

                let mut res = String::from("(");
                res.push_str(&l_res);
                res.push_str(&self.value);
                res.push_str(&r_res);
                res.push_str(")");
                res
            }
            _ => self.value.clone(),
        }
    }

    pub fn get_nth_node(&self, n: usize) -> Child {
        let mut queue: VecDeque<Child> = VecDeque::new();
        queue.push_back(Some(Rc::new(self.clone())));
        let mut cindex = 0;

        let mut current_node: Child = None;
        while !queue.is_empty() {
            current_node = queue.pop_front().unwrap();
            if current_node.is_some() {
                if cindex == n {
                    return current_node;
                }
                cindex += 1;

                if let (Some(l), Some(r)) = (&self.left, &self.right) {
                    queue.push_back(Some(l.clone()));
                    queue.push_back(Some(r.clone()));
                }
            }
        }
        current_node
    }

    pub fn node_count(&self) -> usize {
        if let (Some(l), Some(r)) = (&self.left, &self.right) {
            1 + l.node_count() + r.node_count()
        } else {
            1
        }
    }

    pub fn depth(&self) -> usize {
        if let (Some(l), Some(r)) = (&self.left, &self.right) {
            1 + std::cmp::max(l.depth(), r.depth())
        } else {
            0
        }
    }

    pub fn new(value: String, left: Child, right: Child) -> Child {
        Some(Rc::new(Node { value, left, right }))
    }
}

impl Tree {
    pub fn new(root: Child) -> Self {
        Tree { root }
    }
}

#[derive(Default, Debug, Clone)]
struct GATree {
    inner: Genome<Tree>,
}

impl Mutate for GATree {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        let tree = &self.inner.data;

        fn traverse(root: &Child, config: &MutationConfig, seed: [u8; 32]) -> Child {
            let mut rng: StdRng = SeedableRng::from_seed(seed);

            match root {
                None => None,
                Some(node) => {
                    if rng.gen::<i32>() % 100 < ((100.0 * config.gene_mutation_chance) as i32) {
                        random_node(node.depth(), rng.gen())
                    } else {
                        let new_left = traverse(&node.left, config, rng.gen());
                        let new_right = traverse(&node.right, config, rng.gen());
                        Node::new(node.value.clone(), new_left, new_right)
                    }
                }
            }
        }

        let new_root = traverse(&tree.root, config, seed);
        GATree {
            inner: Genome {
                data: Tree { root: new_root },
                ..Default::default()
            },
        }
    }
}

impl Crossover for GATree {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);

        let tree = self.inner.data.root.clone().unwrap();
        let crossover_tree = &other.inner.data.root.as_ref();

        let cross_point = rng.gen_range(0..tree.node_count());
        let origin_node = tree.get_nth_node(cross_point);
        let cross_node = crossover_tree
            .unwrap()
            .get_nth_node(rng.gen_range(0..crossover_tree.unwrap().node_count()));

        fn traverse(root: &Child, origin_node: &Child, cross_node: &Child) -> Child {
            match (root, origin_node) {
                (None, _) => None,
                (Some(node), None) => {
                    Node::new(node.value.clone(), node.left.clone(), node.right.clone())
                }
                (Some(node), Some(origin)) => {
                    if node.print() == origin.print() {
                        cross_node.clone()
                    } else {
                        let new_left = traverse(&node.left, origin_node, cross_node);
                        let new_right = traverse(&node.right, origin_node, cross_node);

                        Node::new(node.value.clone(), new_left, new_right)
                    }
                }
            }
        }

        let new_root = traverse(&self.inner.data.root, &origin_node, &cross_node);

        let data: Tree = Tree::new(new_root);
        GATree {
            inner: Genome {
                data,
                ..Default::default()
            },
        }
    }
}

impl Generate for GATree {
    fn generate(seed: [u8; 32]) -> Self {
        let root = random_node(MAX_DEPTH, seed);
        GATree {
            inner: Genome {
                data: Tree::new(root),
                ..Default::default()
            },
        }
    }
}

impl FitnessRetrieve for GATree {
    fn get_fitness(&self) -> Option<f64> {
        self.inner.fitness
    }
}

impl Fitness for GATree {
    fn calculate_fitness(&mut self, _seed: [u8; 32]) -> Option<f64> {
        let tree = &self.inner.data;
        let mut wrong: f64 = 0.0;
        (0..10).for_each(|i| {
            (0..10).for_each(|y| match &tree.root {
                None => wrong += 1.0,
                Some(root) => {
                    let actual = root.evaluate(i as f64, y as f64) % (i64::MAX as f64);
                    let real = i * i + y * y;
                    let diff = (real - actual.round() as i64).abs();
                    match diff {
                        0 => {}
                        1..=100 => wrong += 1.0,
                        _ => wrong += 2.0,
                    }
                }
            });
        });
        self.inner.fitness = Some(500.0 - wrong);

        self.inner.fitness
    }
}

pub fn random_node(depth: usize, seed: [u8; 32]) -> Child {
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut val = rng.gen_range(0..7);
    if depth <= 1 {
        val = rng.gen_range(0..2);
    }

    match (val, NR_VARS) {
        (0, _) => Node::new(rng.gen_range(MIN_VALUE..MAX_VALUE).to_string(), None, None),
        (1, 0) => Node::new(rng.gen_range(MIN_VALUE..MAX_VALUE).to_string(), None, None),
        (1, _) => {
            let var = rng.gen_range(0..NR_VARS);
            let mut var_name = "X".to_owned();
            var_name.push_str(var.to_string().as_str());
            Node::new(var_name, None, None)
        }
        (2..=6, _) => {
            let ops = vec!['+', '-', '*', '/', '^'];
            let value = ops[val - 2].to_string();
            let left = random_node(depth - 1, rng.gen());
            let right = random_node(depth - 1, rng.gen());
            Node::new(value, left, right)
        }
        _ => Node::new("borked".to_string(), None, None),
    }
}

fn main() {
    let config = PopulationConfig {
        pop_size: 10,
        crossover_count: 2,
        mutate_count: 2,
        elitism_count: 2,
        mutation_config: MutationConfig {
            gene_mutation_chance: 0.3,
        },
        seed: rand::thread_rng().gen(),
    };
    let mut p: Population<GATree> = Population::new(config);

    (0..1000).for_each(|i| {
        p.tick();
        let best = p.get_best_member();
        println!(
            "Gen {i}: Fitness: {} - {:?}",
            best.get_fitness().unwrap(),
            best.inner.data.root.clone().unwrap().print()
        );
    });
}
