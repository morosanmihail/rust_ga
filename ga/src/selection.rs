use rand::{rngs::StdRng, seq::SliceRandom, Rng};

use crate::traits::SelectionStrategy;

/// Uniform random selection (default). 
#[derive(Debug, Clone, Default)]
pub struct RandomSelection;

/// Pick `size` random candidates, return the one with the best fitness.
#[derive(Debug, Clone)]
pub struct TournamentSelection {
    pub size: usize,
}

/// Fitness-proportional (roulette-wheel) selection.
/// Shifts fitness so all weights are positive before sampling.
#[derive(Debug, Clone, Default)]
pub struct RouletteWheelSelection;

/// Rank-based selection. Best-ranked member gets weight `n`, worst gets `1`.
/// More stable than roulette when fitness values vary widely.
#[derive(Debug, Clone, Default)]
pub struct RankSelection;

impl TournamentSelection {
    pub fn new(size: usize) -> Self {
        TournamentSelection { size }
    }
}

impl SelectionStrategy for RandomSelection {
    fn select(&self, fitnesses: &[Option<f64>], rng: &mut StdRng) -> Option<usize> {
        if fitnesses.is_empty() {
            return None;
        }
        Some(rng.gen_range(0..fitnesses.len()))
    }

    fn select_pair(
        &self,
        fitnesses: &[Option<f64>],
        rng: &mut StdRng,
    ) -> Option<(usize, usize)> {
        let indices: Vec<usize> = (0..fitnesses.len()).collect();
        let picked: Vec<usize> = indices.choose_multiple(rng, 2).copied().collect();
        if picked.len() < 2 {
            return None;
        }
        Some((picked[0], picked[1]))
    }
}

impl SelectionStrategy for TournamentSelection {
    fn select(&self, fitnesses: &[Option<f64>], rng: &mut StdRng) -> Option<usize> {
        let n = fitnesses.len();
        if n == 0 {
            return None;
        }
        let indices: Vec<usize> = (0..n).collect();
        let candidates: Vec<usize> = indices.choose_multiple(rng, self.size).copied().collect();
        candidates.into_iter().max_by(|&a, &b| {
            fitnesses[a]
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&fitnesses[b].unwrap_or(f64::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl SelectionStrategy for RouletteWheelSelection {
    fn select(&self, fitnesses: &[Option<f64>], rng: &mut StdRng) -> Option<usize> {
        let n = fitnesses.len();
        if n == 0 {
            return None;
        }
        let raw: Vec<f64> = fitnesses.iter().map(|f| f.unwrap_or(0.0)).collect();
        let min = raw.iter().cloned().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = raw.iter().map(|&f| f - min + 1.0).collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.gen::<f64>() * total;
        for (i, &w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return Some(i);
            }
        }
        Some(n - 1)
    }
}

impl SelectionStrategy for RankSelection {
    fn select(&self, fitnesses: &[Option<f64>], rng: &mut StdRng) -> Option<usize> {
        let n = fitnesses.len();
        if n == 0 {
            return None;
        }
        // Sort indices best-first
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            fitnesses[b]
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&fitnesses[a].unwrap_or(f64::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Best gets weight n, worst gets weight 1; total = n*(n+1)/2
        let total = n * (n + 1) / 2;
        let mut pick = rng.gen_range(0..total);
        for (rank, &idx) in indices.iter().enumerate() {
            let weight = n - rank;
            if pick < weight {
                return Some(idx);
            }
            pick -= weight;
        }
        indices.last().copied()
    }
}
