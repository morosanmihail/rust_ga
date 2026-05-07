use std::collections::VecDeque;
use super::map::{Map, Structure, Terrain};

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
    Moving { path: Vec<(usize, usize)>, step: usize, for_build: bool },
    Building { tx: usize, ty: usize, structure: Structure, ticks_left: u32 },
}

pub struct Builder {
    pub x: usize,
    pub y: usize,
    pub hp: u32,
    pub instructions: VecDeque<Instruction>,
    /// Index into the original instruction list that is currently executing.
    pub current_instr_idx: Option<usize>,
    state: State,
    instrs_done: usize,
    re_queued: bool,
}

impl Builder {
    pub fn new(x: usize, y: usize, hp: u32) -> Self {
        Builder {
            x, y, hp,
            instructions: VecDeque::new(),
            current_instr_idx: None,
            state: State::Idle,
            instrs_done: 0,
            re_queued: false,
        }
    }

    pub fn push(&mut self, instr: Instruction) {
        self.instructions.push_back(instr);
    }

    pub fn is_alive(&self) -> bool {
        self.hp > 0
    }

    pub fn state_description(&self) -> String {
        match &self.state {
            State::Idle if self.instructions.is_empty() => "idle (done)".into(),
            State::Idle => "idle".into(),
            State::Moving { path, step, .. } => {
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
            State::Moving { path, step, for_build } => {
                let next = step + 1;
                if next < path.len() {
                    self.x = path[next].0;
                    self.y = path[next].1;
                    if next + 1 < path.len() {
                        self.state = State::Moving { path, step: next, for_build };
                    } else if !for_build {
                        // Move instruction arrived — instruction complete.
                        self.instrs_done += 1;
                        self.current_instr_idx = None;
                    }
                    // for_build=true: state stays Idle, process_next picks up re-queued build instr.
                }
            }
            State::Building { tx, ty, structure, ticks_left } => {
                if ticks_left <= 1 {
                    map.get_mut(tx, ty).structure = Some(structure);
                    self.instrs_done += 1;
                    self.current_instr_idx = None;
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
                None => { self.current_instr_idx = None; return; }
            };

            // Re-queued instructions are the same logical instruction — don't advance index.
            let is_new = !self.re_queued;
            self.re_queued = false;
            if is_new {
                self.current_instr_idx = Some(self.instrs_done);
            }

            match instr {
                Instruction::Move(tx, ty) => {
                    if !map.get(tx, ty).is_walkable() { self.instrs_done += 1; continue; }
                    if self.x == tx && self.y == ty   { self.instrs_done += 1; continue; }
                    match map.find_path(self.x, self.y, tx, ty) {
                        Some(path) if path.len() > 1 => {
                            self.state = State::Moving { path, step: 0, for_build: false };
                            return;
                        }
                        _ => { self.instrs_done += 1; continue; }
                    }
                }
                Instruction::BuildTower(tx, ty) => {
                    let cell = map.get(tx, ty);
                    if cell.terrain != Terrain::Plain || cell.structure.is_some() {
                        self.instrs_done += 1; continue;
                    }
                    if self.try_build(map, tx, ty, Structure::Tower { hp: tower_hp }, TOWER_TICKS,
                        || Instruction::BuildTower(tx, ty)) { return; }
                    self.instrs_done += 1;
                }
                Instruction::BuildWall(tx, ty) => {
                    let cell = map.get(tx, ty);
                    if cell.terrain != Terrain::Plain || cell.structure.is_some() {
                        self.instrs_done += 1; continue;
                    }
                    if self.try_build(map, tx, ty, Structure::Wall { hp: wall_hp }, WALL_TICKS,
                        || Instruction::BuildWall(tx, ty)) { return; }
                    self.instrs_done += 1;
                }
                Instruction::BuildBridge(tx, ty) => {
                    let cell = map.get(tx, ty);
                    if cell.terrain != Terrain::Water || cell.structure.is_some() {
                        self.instrs_done += 1; continue;
                    }
                    if self.try_build(map, tx, ty, Structure::Bridge, BRIDGE_TICKS,
                        || Instruction::BuildBridge(tx, ty)) { return; }
                    self.instrs_done += 1;
                }
            }
        }
    }

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
                self.re_queued = true;
                self.state = State::Moving { path, step: 0, for_build: true };
                true
            }
        }
    }
}
