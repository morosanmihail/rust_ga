use ga::{
    population::{MutationConfig, Population, PopulationConfig},
    traits::{Crossover, Fitness, FitnessRetrieve, Generate, Mutate},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Activation functions that can be used by the network.
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Linear, // no non‑linearity
}

impl Activation {
    /// Apply the selected activation element‑wise.
    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
        }
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Tanh
    }
}

#[derive(Default, Debug, Clone)]
struct NeuralNet {
    /// Number of neurons per layer (including input and output).
    pub layers: Vec<usize>,
    /// Weight matrices: `weights[l][i][j]` connects neuron `j` of layer `l`
    /// to neuron `i` of layer `l+1`. Stored row‑major (next‑layer first).
    pub weights: Vec<Vec<Vec<f32>>>,
    /// Bias vectors: one bias per neuron of each *non‑input* layer.
    pub biases: Vec<Vec<f32>>,
    /// Which activation to use for hidden layers (output layer can reuse it
    /// or you can treat it specially in `forward`).
    pub activation: Activation,

    pub fitness: Option<f64>,
}

impl Generate for NeuralNet {
    fn generate(seed: [u8; 32]) -> Self {
        let layers = vec![4, 6, 2]; // TODO: const / param
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for win in layers.windows(2) {
            // Weight matrix for this connection
            let mut w_mat = vec![vec![0.0; win[0]]; win[1]];
            for row in &mut w_mat {
                for v in row.iter_mut() {
                    *v = rng.gen_range(-1.0..=1.0);
                }
            }
            weights.push(w_mat);

            // Bias vector for the *target* layer (win[1] neurons)
            let b_vec: Vec<f32> = (0..win[1]).map(|_| rng.gen_range(-1.0..=1.0)).collect();
            biases.push(b_vec);
        }

        Self {
            layers: layers.to_vec(),
            weights,
            biases,
            ..Default::default()
        }
    }
}

impl Fitness for NeuralNet {
    fn calculate_fitness(&mut self, seed: [u8; 32]) -> Option<f64> {
        pub fn forward(net: &NeuralNet, input: &[f32]) -> Vec<f32> {
            assert_eq!(input.len(), net.layers[0], "Input size mismatch");
            let mut activations = input.to_vec();

            for (layer_idx, (weight_mat, bias_vec)) in
                net.weights.iter().zip(&net.biases).enumerate()
            {
                let next_sz = net.layers[layer_idx + 1];
                let mut next = vec![0.0; next_sz];

                for (neuron_idx, neuron_weights) in weight_mat.iter().enumerate() {
                    // Weighted sum + bias
                    let sum: f32 = neuron_weights
                        .iter()
                        .zip(&activations)
                        .map(|(w, a)| w * a)
                        .sum::<f32>()
                        + bias_vec[neuron_idx];

                    // Apply activation
                    next[neuron_idx] = net.activation.apply(sum);
                }

                activations = next;
            }

            activations
        }

        // TODO: run eval here against some data set / scenario
        let res = forward(&self, &[1.0, 2.0, 3.0, 4.0]);
        self.fitness = Some(0.0);

        self.fitness
    }
}

impl Crossover for NeuralNet {
    fn crossover(&self, other: &Self, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let mut child_weights = Vec::with_capacity(self.weights.len());
        let mut child_biases = Vec::with_capacity(self.biases.len());

        // Weights
        for (wa, wb) in self.weights.iter().zip(&other.weights) {
            let mut layer = Vec::with_capacity(wa.len());
            for (row_a, row_b) in wa.iter().zip(wb) {
                let mut row = Vec::with_capacity(row_a.len());
                for (&va, &vb) in row_a.iter().zip(row_b) {
                    row.push(if rng.gen_bool(0.5) { va } else { vb });
                }
                layer.push(row);
            }
            child_weights.push(layer);
        }

        // Biases
        for (ba, bb) in self.biases.iter().zip(&other.biases) {
            let mut layer = Vec::with_capacity(ba.len());
            for (&va, &vb) in ba.iter().zip(bb) {
                layer.push(if rng.gen_bool(0.5) { va } else { vb });
            }
            child_biases.push(layer);
        }

        Self {
            layers: self.layers.clone(),
            weights: child_weights,
            biases: child_biases,
            activation: self.activation,
            fitness: None,
        }
    }
}

impl Mutate for NeuralNet {
    fn mutate(&self, config: &MutationConfig, seed: [u8; 32]) -> Self {
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let scale = 0.15;
        let rate = 0.5;

        Self {
            layers: self.layers.clone(),
            activation: self.activation,
            weights: self
                .weights
                .iter()
                .map(|layer| {
                    layer
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|w| {
                                    if rng.gen::<f32>() < rate {
                                        (w + rng.sample(rand::distributions::Uniform::new(
                                            -scale, scale,
                                        )))
                                        .clamp(-5.0, 5.0)
                                    } else {
                                        w.clone()
                                    }
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect(),
            biases: self
                .biases
                .iter()
                .map(|bias_vec| {
                    bias_vec
                        .iter()
                        .map(|b| {
                            if rng.gen::<f32>() < rate {
                                (b + rng.sample(rand::distributions::Uniform::new(-scale, scale)))
                                    .clamp(-5.0, 5.0)
                            } else {
                                b.clone()
                            }
                        })
                        .collect()
                })
                .collect(),
            fitness: None,
        }
    }
}

impl FitnessRetrieve for NeuralNet {
    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}

fn main() {
    let config = PopulationConfig {
        pop_size: 100,
        crossover_count: 20,
        mutate_count: 20,
        elitism_count: 4,
        mutation_config: MutationConfig {
            gene_mutation_chance: 0.3,
        },
        seed: rand::thread_rng().gen(),
    };
    let mut p: Population<NeuralNet> = Population::new(config);

    (0..10000).for_each(|i| {
        p.tick();
        let best = p.get_best_member();
        println!(
            "Gen {i}: Fitness: {} ",
            best.get_fitness().unwrap(),
            // best.inner.data.root.clone().unwrap().print()
        );
    });
}
