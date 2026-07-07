use std::num::NonZeroU32;

use clap::Parser;
use cubecl::prelude::*;
use massively::{Executor, MIndex, Zip1, Zip2, op, reduce, transform, util::random};

#[derive(clap::Parser)]
struct Args {
    #[arg(help = "Number of iteration")]
    n: NonZeroU32,
    #[arg(help = "Number of points per iteration")]
    m: NonZeroU32,
}

fn main() -> Result<(), massively::Error> {
    let args = Args::parse();

    let n = args.n.get() as MIndex;
    let m = args.m.get() as MIndex;

    // Setup
    let exec = {
        // let device = cubecl::wgpu::WgpuDevice::Cpu;
        // let device = cubecl::wgpu::WgpuDevice::IntegratedGpu(0);
        // let device = cubecl::wgpu::WgpuDevice::DiscreteGpu(0);
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        // let device = cubecl::hip::device::AmdDevice::new(0);
        // let device = cubecl::cuda::CudaDevice::new(0);

        Executor::<cubecl::wgpu::WgpuRuntime>::new(device)
        // massively::Executor::<cubecl::hip::HipRuntime>::new(device)
        // massively::Executor::<cubecl::cuda::CudaRuntime>::new(device)
    };

    let mut sum_pi = 0.;

    for i in 0..n {
        let seed = i as u64 * 2;
        let x = random::uniform_distribution_f32(&exec, m, 0.0, 1.0, seed)?;
        let y = random::uniform_distribution_f32(&exec, m, 0.0, 1.0, seed + 1)?;
        let hits = exec.full(m, 0_u32)?;

        transform(
            &exec,
            Zip2(x.slice(..), y.slice(..)),
            DetectHit,
            Zip1(hits.slice_mut(..)),
        )?;

        // Count the number of ones.
        let (n_hits,) = reduce(&exec, Zip1(hits.slice(..)), (0_u32,), CountHit)?;

        let pi = (n_hits as f64 / m as f64) * 4.;
        sum_pi += pi;
    }

    println!("pi={}", sum_pi / n as f64);
    Ok(())
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
