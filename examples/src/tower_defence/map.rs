use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug, PartialEq)]
pub enum Terrain {
    Plain,
    Rock,
    Water,
}

#[derive(Clone, Debug)]
pub enum Structure {
    Tower { hp: u32 },
    Wall { hp: u32 },
    Bridge,
}

impl Structure {
    pub fn is_blocking_enemy(&self) -> bool {
        matches!(self, Structure::Tower { .. } | Structure::Wall { .. })
    }

    /// Returns true if the structure was destroyed (hp hit 0). Bridge is indestructible.
    pub fn take_damage(&mut self, amount: u32) -> bool {
        match self {
            Structure::Tower { hp } | Structure::Wall { hp } => {
                *hp = hp.saturating_sub(amount);
                *hp == 0
            }
            Structure::Bridge => false,
        }
    }

    pub fn symbol(&self) -> char {
        match self {
            Structure::Tower { .. } => 'T',
            Structure::Wall { .. } => 'W',
            Structure::Bridge => 'B',
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub terrain: Terrain,
    pub structure: Option<Structure>,
}

impl Cell {
    pub fn new(terrain: Terrain) -> Self {
        Cell { terrain, structure: None }
    }

    /// Builder: plain with no blocking structure, or water+bridge.
    pub fn is_walkable(&self) -> bool {
        match (&self.terrain, &self.structure) {
            (Terrain::Plain, None) => true,
            (Terrain::Water, Some(Structure::Bridge)) => true,
            _ => false,
        }
    }

    /// Enemy: plain with no blocking structure, or water+bridge.
    pub fn is_enemy_walkable(&self) -> bool {
        match (&self.terrain, &self.structure) {
            (Terrain::Plain, None) => true,
            (Terrain::Plain, Some(s)) => !s.is_blocking_enemy(),
            (Terrain::Water, Some(Structure::Bridge)) => true,
            _ => false,
        }
    }

    /// Enemy ignoring towers/walls (used for AI route planning).
    /// Rock and bare water still impassable.
    pub fn is_enemy_walkable_ignore_structures(&self) -> bool {
        match &self.terrain {
            Terrain::Plain => true,
            Terrain::Water => matches!(&self.structure, Some(Structure::Bridge)),
            Terrain::Rock => false,
        }
    }
}

pub struct Map {
    pub width: usize,
    pub height: usize,
    cells: Vec<Vec<Cell>>,
    pub spawn_points: Vec<(usize, usize)>,
}

impl Map {
    pub fn new(cells: Vec<Vec<Cell>>, spawn_points: Vec<(usize, usize)>) -> Self {
        let height = cells.len();
        let width = if height > 0 { cells[0].len() } else { 0 };
        Map { width, height, cells, spawn_points }
    }

    pub fn get(&self, x: usize, y: usize) -> &Cell { &self.cells[y][x] }
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut Cell { &mut self.cells[y][x] }

    pub fn neighbors4(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        if x > 0            { v.push((x - 1, y)); }
        if x + 1 < self.width  { v.push((x + 1, y)); }
        if y > 0            { v.push((x, y - 1)); }
        if y + 1 < self.height { v.push((x, y + 1)); }
        v
    }

    // ── Builder pathfinding ───────────────────────────────────────────────────

    pub fn find_path(&self, sx: usize, sy: usize, tx: usize, ty: usize) -> Option<Vec<(usize, usize)>> {
        bfs(self, sx, sy, |_x, _y, nx, ny| {
            self.cells[ny][nx].is_walkable() || (nx == tx && ny == ty)
        }, |x, y| x == tx && y == ty)
    }

    /// Path ending at any walkable tile adjacent to (tx,ty).
    pub fn find_path_to_adjacent(&self, sx: usize, sy: usize, tx: usize, ty: usize) -> Option<Vec<(usize, usize)>> {
        let targets: HashSet<(usize, usize)> = self.neighbors4(tx, ty)
            .into_iter()
            .filter(|&(nx, ny)| self.cells[ny][nx].is_walkable())
            .collect();
        if targets.is_empty() { return None; }
        if targets.contains(&(sx, sy)) { return Some(vec![(sx, sy)]); }
        bfs(self, sx, sy,
            |_cx, _cy, nx, ny| self.cells[ny][nx].is_walkable(),
            |x, y| targets.contains(&(x, y)))
    }

    // ── Enemy pathfinding ─────────────────────────────────────────────────────

    pub fn find_enemy_path(&self, sx: usize, sy: usize, tx: usize, ty: usize) -> Option<Vec<(usize, usize)>> {
        bfs(self, sx, sy,
            |_, _, nx, ny| self.cells[ny][nx].is_enemy_walkable() || (nx == tx && ny == ty),
            |x, y| x == tx && y == ty)
    }

    pub fn find_enemy_path_ignore_structures(&self, sx: usize, sy: usize, tx: usize, ty: usize) -> Option<Vec<(usize, usize)>> {
        bfs(self, sx, sy,
            |_, _, nx, ny| self.cells[ny][nx].is_enemy_walkable_ignore_structures() || (nx == tx && ny == ty),
            |x, y| x == tx && y == ty)
    }
}

fn bfs<FNeighbor, FGoal>(
    map: &Map,
    sx: usize, sy: usize,
    can_enter: FNeighbor,
    is_goal: FGoal,
) -> Option<Vec<(usize, usize)>>
where
    FNeighbor: Fn(usize, usize, usize, usize) -> bool,
    FGoal: Fn(usize, usize) -> bool,
{
    if is_goal(sx, sy) { return Some(vec![(sx, sy)]); }
    let mut came_from: HashMap<(usize, usize), Option<(usize, usize)>> = HashMap::new();
    let mut queue = VecDeque::new();
    came_from.insert((sx, sy), None);
    queue.push_back((sx, sy));
    while let Some((cx, cy)) = queue.pop_front() {
        for (nx, ny) in map.neighbors4(cx, cy) {
            if came_from.contains_key(&(nx, ny)) { continue; }
            if !can_enter(cx, cy, nx, ny) { continue; }
            came_from.insert((nx, ny), Some((cx, cy)));
            if is_goal(nx, ny) { return Some(reconstruct(&came_from, (nx, ny))); }
            queue.push_back((nx, ny));
        }
    }
    None
}

fn reconstruct(
    came_from: &HashMap<(usize, usize), Option<(usize, usize)>>,
    end: (usize, usize),
) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    let mut cur = end;
    loop {
        path.push(cur);
        match came_from[&cur] {
            None => break,
            Some(prev) => cur = prev,
        }
    }
    path.reverse();
    path
}
