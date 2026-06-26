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
        let xs = massively::util::random::uniform_distribution_f32(&exec, m as usize, i as u64).unwrap();
        let ys = massively::util::random::uniform_distribution_f32(&exec, m as usize, (i+1) as u64).unwrap();

        // Within the quarter circle -> 1, otherwise -> 0.
        let mut hits = exec.filled(m as usize, 0).unwrap();
        massively::transform(&exec, SoA2(xs.slice(..), ys.slice(..)), DetectHit, SoA1(hits.slice_mut(..))).unwrap();

        // Count the number of ones.
        let (n_hits,) = massively::reduce(&exec, SoA1(hits.slice(..)), (0,), CountHit).unwrap();

        let pi = (n_hits as f64 / m as f64) * 4.;
        sum_pi += pi;
    }

    println!("pi={}", sum_pi / n as f64)
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
