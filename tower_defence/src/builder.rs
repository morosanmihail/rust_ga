use std::collections::VecDeque;
use crate::map::{Map, Structure, Terrain};

#[derive(Clone, Debug)]
pub enum Instruction {
    BuildTower(usize, usize),
    BuildWall(usize, usize),
    BuildBridge(usize, usize),
    Move(usize, usize),
}

const TOWER_TICKS: u32 = 3;
const WALL_TICKS: u32 = 2;
const BRIDGE_TICKS: u32 = 4;

#[derive(Clone, Debug)]
enum State {
    Idle,
    Moving { path: Vec<(usize, usize)>, step: usize },
    Building { tx: usize, ty: usize, structure: Structure, ticks_left: u32 },
}

pub struct Builder {
    pub x: usize,
    pub y: usize,
    pub hp: u32,
    pub instructions: VecDeque<Instruction>,
    state: State,
}

impl Builder {
    pub fn new(x: usize, y: usize, hp: u32) -> Self {
        Builder { x, y, hp, instructions: VecDeque::new(), state: State::Idle }
    }

    pub fn push(&mut self, instr: Instruction) {
        self.instructions.push_back(instr);
    }

    pub fn is_idle(&self) -> bool {
        matches!(self.state, State::Idle) && self.instructions.is_empty()
    }

    pub fn is_alive(&self) -> bool {
        self.hp > 0
    }

    pub fn state_description(&self) -> String {
        match &self.state {
            State::Idle if self.instructions.is_empty() => "idle (done)".into(),
            State::Idle => "idle".into(),
            State::Moving { path, step } => {
                let d = path.last().unwrap();
                format!("moving to ({},{}) step {}/{}", d.0, d.1, step, path.len() - 1)
            }
            State::Building { tx, ty, structure, ticks_left } => {
                format!("building {} at ({},{}) — {} ticks left", structure.symbol(), tx, ty, ticks_left)
            }
        }
    }

    pub fn tick(&mut self, map: &mut Map, tower_hp: u32, wall_hp: u32) {
        let state = std::mem::replace(&mut self.state, State::Idle);
        match state {
            State::Idle => self.process_next(map, tower_hp, wall_hp),
            State::Moving { path, step } => {
                let next = step + 1;
                if next < path.len() {
                    self.x = path[next].0;
                    self.y = path[next].1;
                    if next + 1 < path.len() {
                        self.state = State::Moving { path, step: next };
                    }
                }
            }
            State::Building { tx, ty, structure, ticks_left } => {
                if ticks_left <= 1 {
                    map.get_mut(tx, ty).structure = Some(structure);
                } else {
                    self.state = State::Building { tx, ty, structure, ticks_left: ticks_left - 1 };
                }
            }
        }
    }

    fn process_next(&mut self, map: &Map, tower_hp: u32, wall_hp: u32) {
        loop {
            let instr = match self.instructions.pop_front() {
                Some(i) => i,
                None => return,
            };
            match instr {
                Instruction::Move(tx, ty) => {
                    if !map.get(tx, ty).is_walkable() { continue; }
                    if self.x == tx && self.y == ty { continue; }
                    match map.find_path(self.x, self.y, tx, ty) {
                        Some(path) if path.len() > 1 => {
                            self.state = State::Moving { path, step: 0 };
                            return;
                        }
                        _ => continue,
                    }
                }
                Instruction::BuildTower(tx, ty) => {
                    let cell = map.get(tx, ty);
                    if cell.terrain != Terrain::Plain || cell.structure.is_some() { continue; }
                    if self.try_build(map, tx, ty, Structure::Tower { hp: tower_hp }, TOWER_TICKS,
                        || Instruction::BuildTower(tx, ty)) { return; }
                }
                Instruction::BuildWall(tx, ty) => {
                    let cell = map.get(tx, ty);
                    if cell.terrain != Terrain::Plain || cell.structure.is_some() { continue; }
                    if self.try_build(map, tx, ty, Structure::Wall { hp: wall_hp }, WALL_TICKS,
                        || Instruction::BuildWall(tx, ty)) { return; }
                }
                Instruction::BuildBridge(tx, ty) => {
                    let cell = map.get(tx, ty);
                    if cell.terrain != Terrain::Water || cell.structure.is_some() { continue; }
                    if self.try_build(map, tx, ty, Structure::Bridge, BRIDGE_TICKS,
                        || Instruction::BuildBridge(tx, ty)) { return; }
                }
            }
        }
    }

    /// Returns true if action was taken (move or build started), false if unreachable (caller skips).
    fn try_build<F>(&mut self, map: &Map, tx: usize, ty: usize, structure: Structure, ticks: u32, make_instr: F) -> bool
    where F: Fn() -> Instruction
    {
        match map.find_path_to_adjacent(self.x, self.y, tx, ty) {
            None => false,
            Some(path) if path.len() == 1 => {
                self.state = State::Building { tx, ty, structure, ticks_left: ticks };
                true
            }
            Some(path) => {
                self.instructions.push_front(make_instr());
                self.state = State::Moving { path, step: 0 };
                true
            }
        }
    }
}
