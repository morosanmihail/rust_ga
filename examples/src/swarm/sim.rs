pub const WORLD_W: f64 = 60.0;
pub const WORLD_H: f64 = 20.0;
pub const GOAL_X: f64 = 56.0;

const AGENT_SPEED: f64 = 0.7;
const OBS_INFLUENCE: f64 = 5.5;
const NBR_RANGE: f64 = 7.0;
const SEP_RANGE: f64 = 2.0;
const ACCEL_DT: f64 = 0.35;

#[derive(Clone, Debug)]
pub struct Agent {
    pub pos: (f64, f64),
    pub vel: (f64, f64),
    pub reached_goal: bool,
}

#[derive(Clone, Debug)]
pub struct Obstacle {
    pub pos: (f64, f64),
    pub radius: f64,
}

pub struct SteeringWeights {
    pub goal_pull: f64,
    pub obstacle_push: f64,
    pub neighbor_align: f64,
    pub neighbor_separate: f64,
}

#[derive(Clone)]
pub struct SwarmSim {
    pub agents: Vec<Agent>,
    pub obstacles: Vec<Obstacle>,
    pub tick: u64,
}

impl SwarmSim {
    pub fn new() -> Self {
        let n = 15usize;
        let agents = (0..n)
            .map(|i| {
                let y = 1.5 + i as f64 * (WORLD_H - 3.0) / (n - 1) as f64;
                Agent {
                    pos: (2.0, y),
                    vel: (AGENT_SPEED * 0.5, 0.0),
                    reached_goal: false,
                }
            })
            .collect();

        // Two barrier gates, each with three obstacles.
        // Center obstacle at y=9.0 (not 10.0) so the y=10 agent has dy=1.0 →
        // nonzero lateral repulsion force → escapes toward lower corridor.
        // Corridors per gate: y≈3-5.5 (upper) and y≈12.5-17 (lower).
        let obstacles = vec![
            Obstacle { pos: (22.0, 0.0), radius: 3.0 },
            Obstacle { pos: (22.0, 9.0), radius: 3.5 },
            Obstacle { pos: (22.0, 20.0), radius: 3.0 },
            Obstacle { pos: (42.0, 0.0), radius: 3.0 },
            Obstacle { pos: (42.0, 9.0), radius: 3.5 },
            Obstacle { pos: (42.0, 20.0), radius: 3.0 },
        ];

        SwarmSim { agents, obstacles, tick: 0 }
    }

    pub fn step(&mut self, w: &SteeringWeights) {
        let prev_pos: Vec<_> = self.agents.iter().map(|a| a.pos).collect();
        let prev_vel: Vec<_> = self.agents.iter().map(|a| a.vel).collect();

        for (i, agent) in self.agents.iter_mut().enumerate() {
            if agent.reached_goal {
                continue;
            }

            // Goal: constant pull to the right.
            let fx_goal = w.goal_pull;
            let fy_goal = 0.0_f64;

            // Obstacle repulsion (inverse-square with influence radius).
            let (mut fx_obs, mut fy_obs) = (0.0_f64, 0.0_f64);
            for obs in &self.obstacles {
                let dx = agent.pos.0 - obs.pos.0;
                let dy = agent.pos.1 - obs.pos.1;
                let dist = dx.hypot(dy).max(0.01);
                if dist < obs.radius + OBS_INFLUENCE {
                    let strength = w.obstacle_push / dist.powi(2);
                    fx_obs += (dx / dist) * strength;
                    fy_obs += (dy / dist) * strength;
                }
            }

            // Neighbor alignment: average velocity of nearby agents.
            let (mut fx_align, mut fy_align) = (0.0_f64, 0.0_f64);
            let mut n_align = 0usize;
            for (j, &pos) in prev_pos.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = (agent.pos.0 - pos.0).hypot(agent.pos.1 - pos.1);
                if d < NBR_RANGE {
                    fx_align += prev_vel[j].0;
                    fy_align += prev_vel[j].1;
                    n_align += 1;
                }
            }
            if n_align > 0 {
                fx_align = fx_align / n_align as f64 * w.neighbor_align;
                fy_align = fy_align / n_align as f64 * w.neighbor_align;
            }

            // Separation: push away from agents within personal space.
            let (mut fx_sep, mut fy_sep) = (0.0_f64, 0.0_f64);
            for (j, &pos) in prev_pos.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = agent.pos.0 - pos.0;
                let dy = agent.pos.1 - pos.1;
                let d = dx.hypot(dy).max(0.01);
                if d < SEP_RANGE {
                    fx_sep += (dx / d) / d * w.neighbor_separate;
                    fy_sep += (dy / d) / d * w.neighbor_separate;
                }
            }

            let fx = fx_goal + fx_obs + fx_align + fx_sep;
            let fy = fy_goal + fy_obs + fy_align + fy_sep;

            let nvx = (agent.vel.0 + fx * ACCEL_DT).clamp(-AGENT_SPEED, AGENT_SPEED);
            let nvy = (agent.vel.1 + fy * ACCEL_DT).clamp(-AGENT_SPEED, AGENT_SPEED);

            let spd = nvx.hypot(nvy);
            let (nvx, nvy) = if spd > AGENT_SPEED {
                (nvx / spd * AGENT_SPEED, nvy / spd * AGENT_SPEED)
            } else {
                (nvx, nvy)
            };

            agent.vel = (nvx, nvy);

            let new_x = agent.pos.0 + nvx;
            let new_y = agent.pos.1 + nvy;

            // Hard collision with obstacles: reject move if inside any obstacle.
            let inside_obs = self.obstacles.iter().any(|obs| {
                (new_x - obs.pos.0).hypot(new_y - obs.pos.1) < obs.radius
            });

            if inside_obs {
                agent.vel = (agent.vel.0 * -0.3, agent.vel.1 * -0.3);
            } else {
                if new_x < 0.0 {
                    agent.pos.0 = 0.0;
                    agent.vel.0 = 0.0;
                } else {
                    agent.pos.0 = new_x.min(WORLD_W);
                }
                if new_y < 0.0 {
                    agent.pos.1 = 0.0;
                    agent.vel.1 = 0.0;
                } else if new_y >= WORLD_H {
                    agent.pos.1 = WORLD_H - 0.01;
                    agent.vel.1 = 0.0;
                } else {
                    agent.pos.1 = new_y;
                }
            }

            if agent.pos.0 >= GOAL_X {
                agent.reached_goal = true;
            }
        }

        self.tick += 1;
    }

    pub fn progress_score(&self) -> f64 {
        self.agents
            .iter()
            .map(|a| if a.reached_goal { WORLD_W } else { a.pos.0 })
            .sum::<f64>()
            / self.agents.len() as f64
    }

    pub fn agents_at_goal(&self) -> usize {
        self.agents.iter().filter(|a| a.reached_goal).count()
    }
}
