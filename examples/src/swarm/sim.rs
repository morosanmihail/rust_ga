use rand::{rngs::StdRng, Rng, SeedableRng};

pub const WORLD_W: f64 = 60.0;
pub const WORLD_H: f64 = 20.0;

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

#[derive(Clone, Debug)]
pub struct GoalRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl GoalRect {
    pub fn contains(&self, pos: (f64, f64)) -> bool {
        pos.0 >= self.x && pos.0 <= self.x + self.w
            && pos.1 >= self.y && pos.1 <= self.y + self.h
    }

    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
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
    pub goal: GoalRect,
    pub tick: u64,
}

impl SwarmSim {
    fn make_agents() -> Vec<Agent> {
        let n = 15usize;
        (0..n)
            .map(|i| {
                let y = 1.5 + i as f64 * (WORLD_H - 3.0) / (n - 1) as f64;
                Agent { pos: (2.0, y), vel: (AGENT_SPEED * 0.5, 0.0), reached_goal: false }
            })
            .collect()
    }

    pub fn new() -> Self {
        let obstacles = vec![
            Obstacle { pos: (22.0, 0.0),  radius: 3.0 },
            Obstacle { pos: (22.0, 9.0),  radius: 3.5 },
            Obstacle { pos: (22.0, 20.0), radius: 3.0 },
            Obstacle { pos: (42.0, 0.0),  radius: 3.0 },
            Obstacle { pos: (42.0, 9.0),  radius: 3.5 },
            Obstacle { pos: (42.0, 20.0), radius: 3.0 },
        ];
        let goal = GoalRect { x: 50.0, y: 8.0, w: 4.0, h: 4.0 };
        SwarmSim { agents: Self::make_agents(), obstacles, goal, tick: 0 }
    }

    /// Random 1–3 barrier gates + a random goal rectangle on the right side.
    /// Mid-obstacle y stays in [6.5, 9.5] — avoids aligning with agent at y=10.
    pub fn new_random(seed: u64) -> Self {
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        let n_gates: usize = rng.gen_range(1..=3);
        let mut obstacles = Vec::new();
        for g in 0..n_gates {
            let x = 12.0 + (g as f64 + rng.gen_range(0.2f64..0.8)) / n_gates as f64 * 34.0;
            let mid_y: f64 = rng.gen_range(6.5..9.5);
            obstacles.push(Obstacle { pos: (x, 0.0),     radius: rng.gen_range(2.5..3.5) });
            obstacles.push(Obstacle { pos: (x, mid_y),   radius: rng.gen_range(3.0..4.5) });
            obstacles.push(Obstacle { pos: (x, WORLD_H), radius: rng.gen_range(2.5..3.5) });
        }

        let gw: f64 = rng.gen_range(3.0..5.0);
        let gh: f64 = rng.gen_range(3.0..5.0);
        let gx: f64 = rng.gen_range(48.0_f64..(WORLD_W - gw - 1.0));
        let gy: f64 = rng.gen_range(1.0_f64..(WORLD_H - gh - 1.0));
        let goal = GoalRect { x: gx, y: gy, w: gw, h: gh };

        SwarmSim { agents: Self::make_agents(), obstacles, goal, tick: 0 }
    }

    pub fn step(&mut self, w: &SteeringWeights) {
        let prev_pos: Vec<_> = self.agents.iter().map(|a| a.pos).collect();
        let prev_vel: Vec<_> = self.agents.iter().map(|a| a.vel).collect();
        let gc = self.goal.center();

        for (i, agent) in self.agents.iter_mut().enumerate() {
            if agent.reached_goal {
                continue;
            }

            // Goal: pull toward goal center.
            let gdx = gc.0 - agent.pos.0;
            let gdy = gc.1 - agent.pos.1;
            let gdist = gdx.hypot(gdy).max(0.01);
            let fx_goal = (gdx / gdist) * w.goal_pull;
            let fy_goal = (gdy / gdist) * w.goal_pull;

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
                if i == j { continue; }
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
                if i == j { continue; }
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

            if self.goal.contains(agent.pos) {
                agent.reached_goal = true;
            }
        }

        self.tick += 1;
    }

    pub fn progress_score(&self) -> f64 {
        let goal_cx = self.goal.center().0;
        self.agents
            .iter()
            .map(|a| if a.reached_goal { goal_cx } else { a.pos.0.min(goal_cx) })
            .sum::<f64>()
            / self.agents.len() as f64
    }

    pub fn agents_at_goal(&self) -> usize {
        self.agents.iter().filter(|a| a.reached_goal).count()
    }
}
