use std::collections::HashMap;
use crate::builder::Builder;
use crate::enemy::{Enemy, EnemyAction};
use crate::map::Map;

pub struct Config {
    /// Ticks before first enemy spawn.
    pub spawn_delay: u64,
    /// Ticks between subsequent spawn waves.
    pub spawn_interval: u64,
    /// Enemies spawned per spawn point per wave.
    pub enemies_per_spawn: usize,
    pub enemy_hp: u32,
    /// Damage each enemy deals per tick (to builder or structure).
    pub enemy_damage: u32,
    /// Euclidean range of towers.
    pub tower_range: f64,
    /// Damage per tower shot.
    pub tower_damage: u32,
    pub tower_hp: u32,
    pub wall_hp: u32,
    pub builder_hp: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            spawn_delay: 100,
            spawn_interval: 10,
            enemies_per_spawn: 1,
            enemy_hp: 10,
            enemy_damage: 2,
            tower_range: 4.0,
            tower_damage: 3,
            tower_hp: 20,
            wall_hp: 15,
            builder_hp: 50,
        }
    }
}

pub struct Simulation {
    pub map: Map,
    pub builder: Builder,
    pub enemies: Vec<Enemy>,
    pub tick: u64,
    pub config: Config,
    pub enemies_killed: u32,
    next_enemy_id: u32,
}

impl Simulation {
    pub fn new(map: Map, builder: Builder, config: Config) -> Self {
        Simulation { map, builder, enemies: Vec::new(), tick: 0, config, enemies_killed: 0, next_enemy_id: 0 }
    }

    pub fn builder_done(&self) -> bool { self.builder.is_idle() }
    pub fn is_game_over(&self) -> bool { !self.builder.is_alive() }

    pub fn tick(&mut self) {
        // 1. Builder acts.
        self.builder.tick(&mut self.map, self.config.tower_hp, self.config.wall_hp);

        // 2. Spawn enemies.
        if self.tick >= self.config.spawn_delay {
            let elapsed = self.tick - self.config.spawn_delay;
            if elapsed % self.config.spawn_interval == 0 {
                let spawns: Vec<(usize, usize)> = self.map.spawn_points.clone();
                for (sx, sy) in spawns {
                    for _ in 0..self.config.enemies_per_spawn {
                        self.enemies.push(Enemy::new(
                            self.next_enemy_id, sx, sy, self.config.enemy_hp,
                        ));
                        self.next_enemy_id += 1;
                    }
                }
            }
        }

        // 3. Towers shoot: each tower picks the closest live enemy in range and damages it.
        let mut tower_positions: Vec<(usize, usize)> = Vec::new();
        for y in 0..self.map.height {
            for x in 0..self.map.width {
                if let Some(s) = &self.map.get(x, y).structure {
                    if matches!(s, crate::map::Structure::Tower { .. }) {
                        tower_positions.push((x, y));
                    }
                }
            }
        }
        for (tx, ty) in tower_positions {
            let range = self.config.tower_range;
            let dmg = self.config.tower_damage;
            // Find closest live enemy in range.
            let target = self.enemies.iter_mut()
                .filter(|e| e.is_alive())
                .filter(|e| {
                    let dx = e.x as f64 - tx as f64;
                    let dy = e.y as f64 - ty as f64;
                    (dx * dx + dy * dy).sqrt() <= range
                })
                .min_by(|a, b| {
                    let da = dist(tx, ty, a.x, a.y);
                    let db = dist(tx, ty, b.x, b.y);
                    da.partial_cmp(&db).unwrap()
                });
            if let Some(enemy) = target {
                enemy.hp = enemy.hp.saturating_sub(dmg);
            }
        }

        // 4. Enemies decide actions (read-only map + builder position).
        let bx = self.builder.x;
        let by = self.builder.y;
        let actions: Vec<EnemyAction> = self.enemies.iter()
            .map(|e| if e.is_alive() { e.decide_action(&self.map, bx, by) } else { EnemyAction::Wait })
            .collect();

        // 5. Apply enemy actions.
        let mut builder_damage: u32 = 0;
        let mut structure_damage: HashMap<(usize, usize), u32> = HashMap::new();

        for (i, action) in actions.into_iter().enumerate() {
            match action {
                EnemyAction::Move(nx, ny) => {
                    self.enemies[i].x = nx;
                    self.enemies[i].y = ny;
                }
                EnemyAction::AttackBuilder => {
                    builder_damage += self.config.enemy_damage;
                }
                EnemyAction::AttackStructure(x, y) => {
                    *structure_damage.entry((x, y)).or_insert(0) += self.config.enemy_damage;
                }
                EnemyAction::Despawn => {
                    self.enemies[i].hp = 0;
                }
                EnemyAction::Wait => {}
            }
        }

        self.builder.hp = self.builder.hp.saturating_sub(builder_damage);

        for ((x, y), dmg) in structure_damage {
            if let Some(cell) = Some(self.map.get_mut(x, y)) {
                if let Some(structure) = &mut cell.structure {
                    if structure.take_damage(dmg) {
                        cell.structure = None;
                    }
                }
            }
        }

        // 6. Remove dead enemies, count kills.
        let before = self.enemies.len();
        self.enemies.retain(|e| e.is_alive());
        self.enemies_killed += (before - self.enemies.len()) as u32;

        self.tick += 1;
    }
}

fn dist(ax: usize, ay: usize, bx: usize, by: usize) -> f64 {
    let dx = ax as f64 - bx as f64;
    let dy = ay as f64 - by as f64;
    (dx * dx + dy * dy).sqrt()
}
