# pi-monte-carlo

A GPU-accelerated example that estimates pi with the Monte Carlo method using
[Massively v0.74](https://github.com/akiradeveloper/massively) and
[CubeCL](https://github.com/tracel-ai/cubecl).

## How it works

The program generates uniformly distributed points in `[0, 1) × [0, 1)` and
counts how many of them are at most one unit away from the origin. Each
iteration estimates pi as follows:

```text
pi ≈ 4 × (number of points inside the unit circle) / (total number of points)
```

It repeats this calculation for the requested number of iterations and prints
the average. The random sequences, coordinate zip, and hit detection use
Massively's lazy iterators, allowing the program to reduce the input without
materializing every point in GPU memory.

## Requirements

- Rust with Edition 2024 support (tested with Rust 1.96.0)
- A GPU or graphics adapter and drivers supported by CubeCL/wgpu
- `just` (optional; required only for the `justfile` command)

On the first build, Cargo downloads Massively and CubeCL from their Git
repositories.

## Usage

```console
cargo run --release -- <iterations> <points-per-iteration>
```

Both arguments must be between `1` and the maximum `u32` value. For example,
the following command runs two iterations with 100,000 points each:

```console
$ cargo run --release -- 2 100000
[src/main.rs:49:9] pi = 3.13096
[src/main.rs:49:9] pi = 3.1426
pi=3.13678
```

The estimate from each iteration is written to standard error. The average of
all iterations is written to standard output.

Run the following command to display the CLI help:

```console
cargo run -- --help
```

## Using just

`just run <iterations>` runs a release build with four billion points per
iteration:

```console
just run 10
```

This is a very large workload. Start with `cargo run` and a smaller point count
to verify that the program works on your system.

## GPU backends

By default, the program uses `WgpuRuntime` with
`WgpuDevice::DefaultDevice`. The setup section in `src/main.rs` also contains
commented examples for selecting a specific wgpu adapter or switching to HIP
or CUDA. Enable the matching device and `Executor` combination for the backend
you want to use.

## License

[MIT License](LICENSE)
