use super::map::Map;

#[derive(Debug)]
pub struct Enemy {
    pub id: u32,
    pub x: usize,
    pub y: usize,
    pub hp: u32,
}

#[derive(Debug)]
pub enum EnemyAction {
    Move(usize, usize),
    AttackBuilder,
    AttackStructure(usize, usize),
    Despawn,
    Wait,
}

impl Enemy {
    pub fn new(id: u32, x: usize, y: usize, hp: u32) -> Self {
        Enemy { id, x, y, hp }
    }

    pub fn is_alive(&self) -> bool {
        self.hp > 0
    }

    pub fn decide_action(&self, map: &Map, builder_x: usize, builder_y: usize) -> EnemyAction {
        let builder_pos = (builder_x, builder_y);

        match map.find_enemy_path(self.x, self.y, builder_x, builder_y) {
            Some(path) => {
                if path.len() < 2 {
                    // At same position as builder.
                    return EnemyAction::AttackBuilder;
                }
                let next = path[1];
                if next == builder_pos {
                    EnemyAction::AttackBuilder
                } else {
                    EnemyAction::Move(next.0, next.1)
                }
            }
            None => {
                // No direct path. Try ignoring structures.
                match map.find_enemy_path_ignore_structures(self.x, self.y, builder_x, builder_y) {
                    None => EnemyAction::Despawn,
                    Some(path) => {
                        for &(x, y) in &path[1..] {
                            if let Some(s) = &map.get(x, y).structure {
                                if s.is_blocking_enemy() {
                                    return EnemyAction::AttackStructure(x, y);
                                }
                            }
                        }
                        // Path exists ignoring structures but no blocking structure found —
                        // can happen if builder moved; treat as wait.
                        EnemyAction::Wait
                    }
                }
            }
        }
    }
}
