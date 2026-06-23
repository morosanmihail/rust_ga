# rust_ga

A Rust genetic algorithm library and example workspace.

## Workspace layout

- **`ga/`** — core GA library. Generic population/evolution engine built on
  `Crossover`, `Mutate`, `Fitness`, `FitnessRetrieve`, and `Generate` traits.
  Supports elitism, mutation, crossover, and parallel fitness evaluation
  (`tick_parallel`, via `rayon`).
- **`examples/`** — runnable binaries demonstrating the `ga` crate:
  - `basic_integer_array` — simple integer-array genome example.
  - `equation` — evolves coefficients to fit an equation.

## GA features

- **Deterministic runs** — `Population` is driven by a single `[u8; 32]` seed
  (`StdRng`). Every `tick()` reseeds children RNGs from that seed and rolls a
  fresh seed for the next generation, so a run is fully reproducible given
  the same starting seed and config.
- **Parallel compute** — `tick_parallel()` evaluates each member's fitness
  concurrently with `rayon` (`par_iter_mut`), each member getting its own RNG
  clone derived from the same per-tick seed. Verified to produce results
  identical to the serial `tick()` (see `test_deterministic_parallel_tick`).
- **Import / export of state** — `Population`, `PopulationConfig`, `Genome`,
  and `MutationConfig` all derive `Serialize`/`Deserialize` (`serde_json`).
  A population can be dumped to JSON mid-run and restored later, continuing
  evolution byte-for-byte identically (see `test_deterministic`).
- **Elitism / mutation / crossover / fresh blood mix** — each generation is
  rebuilt from `elitism_count` top performers carried over unchanged,
  `mutate_count` mutated members, `crossover_count` crossed members, and the
  remainder freshly generated to fill `pop_size`.
- **Preseeded populations** — `PopulationConfig::preseeded_population` lets
  you seed initial members directly (e.g. resume from a saved generation, or
  inject hand-crafted genomes) instead of generating the whole population
  randomly.
- **Generic over genome type** — any type implementing `Crossover`, `Mutate`,
  `Fitness`, `FitnessRetrieve`, and `Generate` can be evolved; the `examples`
  crates plug in their own genome types.

## Build & run

```sh
cargo build
cargo run --bin basic_integer_array
cargo run --bin equation
```

## Tests

```sh
cargo test
```
