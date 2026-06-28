use clap::Parser;
use cubecl::prelude::*;
use massively::{op, prelude::*};

#[derive(clap::Parser)]
struct Args {
    #[arg(help = "Number of iteration")]
    n: u32,
    #[arg(help = "Number of points per iteration")]
    m: u32,
}

fn main() {
    let args = Args::parse();

    let n = args.n;
    let m = args.m;

    // Setup
    let exec = {
        // let device = cubecl::wgpu::WgpuDevice::Cpu;
        // let device = cubecl::wgpu::WgpuDevice::IntegratedGpu(0);
        // let device = cubecl::wgpu::WgpuDevice::DiscreteGpu(0);
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        // let device = cubecl::hip::device::AmdDevice::new(0);
        // let device = cubecl::cuda::CudaDevice::new(0);

        massively::Executor::<cubecl::wgpu::WgpuRuntime>::new(device)
        // massively::Executor::<cubecl::hip::HipRuntime>::new(device)
        // massively::Executor::<cubecl::cuda::CudaRuntime>::new(device)
    };

    let mut sum_pi = 0.;

    for i in 0..n {
        let seed1 = massively::slice::constant_slice(m as usize, i as u64);
        let seed2 = massively::slice::constant_slice(m as usize, (i+1) as u64);

        let idx = massively::slice::tabulate_slice(m as usize);


        let x = massively::slice::transform_slice(SoA2(seed1, idx), RandomF32);
        let y = massively::slice::transform_slice(SoA2(seed2, idx), RandomF32);

        // Within the quarter circle -> 1, otherwise -> 0.
        let hits = massively::slice::transform_slice(SoA2(x,y), DetectHit);

        // Count the number of ones.
        let (n_hits,) = massively::reduce(&exec, SoA1(hits), (0,), CountHit).unwrap();

        let pi = (n_hits as f64 / m as f64) * 4.;
        sum_pi += pi;
    }

    println!("pi={}", sum_pi / n as f64)
}

struct RandomF32;
#[cubecl::cube]
impl<B> op::UnaryOp<B, (u64, u32)> for RandomF32 where B: cubecl::Runtime {
    type Output = (f32,);

    fn apply(inp: (u64, u32)) -> (f32,) {
        let (seed, i) = inp;
        let x= massively::util::random::uniform_f32(seed, i);
        (x,)
    }
}

struct DetectHit;
#[cubecl::cube]
impl<B> op::UnaryOp<B, (f32, f32)> for DetectHit
where
    B: cubecl::Runtime,
{
    type Output = (u32,);

    fn apply(p: (f32, f32)) -> (u32,) {
        let (x, y) = p;
        let d2 = x * x + y * y;
        if d2 <= 1. { (1u32,) } else { (0u32,) }
    }
}

struct CountHit;
#[cubecl::cube]
impl<B> op::ReductionOp<B, (u32,)> for CountHit
where
    B: cubecl::Runtime,
{
    fn apply(x: (u32,), y: (u32,)) -> (u32,) {
        (x.0 + y.0,)
    }
}
