use std::collections::HashMap;
use crate::map::Terrain;
use crate::simulation::Simulation;

/// ASCII render of current simulation state.
///
/// Cell priority:
///   1. builder `@`
///   2. enemies: `E` (1), `2`..`9`, `+` (10+)
///   3. structure symbol (T/W/B)
///   4. spawn `S`
///   5. terrain (`.` `#` `~`)
pub fn render(sim: &Simulation) {
    // Build enemy count map.
    let mut enemy_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for e in &sim.enemies {
        *enemy_counts.entry((e.x, e.y)).or_insert(0) += 1;
    }

    println!("┌── Tick {} ──────────────────────────────┐", sim.tick);

    for y in 0..sim.map.height {
        let mut row = String::with_capacity(sim.map.width);
        for x in 0..sim.map.width {
            // Builder
            if x == sim.builder.x && y == sim.builder.y {
                row.push('@');
                continue;
            }
            // Enemies
            if let Some(&n) = enemy_counts.get(&(x, y)) {
                let ch = match n {
                    1 => 'E',
                    2..=9 => char::from_digit(n as u32, 10).unwrap(),
                    _ => '+',
                };
                row.push(ch);
                continue;
            }
            let is_spawn = sim.map.spawn_points.contains(&(x, y));
            let cell = sim.map.get(x, y);
            let ch = match &cell.structure {
                Some(s) => s.symbol(),
                None => match &cell.terrain {
                    Terrain::Plain => if is_spawn { 'S' } else { '.' },
                    Terrain::Rock  => '#',
                    Terrain::Water => '~',
                },
            };
            row.push(ch);
        }
        println!("{}", row);
    }

    // Status
    println!("  builder ({},{}) HP:{} — {}",
        sim.builder.x, sim.builder.y, sim.builder.hp, sim.builder.state_description());
    println!("  queued: {}  enemies alive: {}  spawns: {:?}",
        sim.builder.instructions.len(), sim.enemies.len(), sim.map.spawn_points);
    if !sim.enemies.is_empty() {
        let preview: Vec<String> = sim.enemies.iter().take(8)
            .map(|e| format!("E{}@({},{})hp{}", e.id, e.x, e.y, e.hp))
            .collect();
        println!("  [{}{}]", preview.join(" "),
            if sim.enemies.len() > 8 { format!(" +{}", sim.enemies.len() - 8) } else { String::new() });
    }
    println!();
}

