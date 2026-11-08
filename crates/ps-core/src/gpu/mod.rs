//! GPU compute through wgpu: Metal on macOS (Intel and Apple Silicon), DirectX 12 or Vulkan on
//! Windows, Vulkan on Linux. Runs Richardson–Lucy with a Gaussian PSF (here) and the filters
//! the other modules are built from ([`ops`]): starlet scales, Gaussian blur, bilateral filter,
//! local minimum/maximum. Always optional: without a usable GPU, for small images, or if
//! anything fails, the CPU path runs instead and gives the same result.
//!
//! The device is created once, lazily, apart from the one drawing the interface.

mod ops;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

/// Threads per workgroup of the element-wise kernels.
const GROUP: u32 = 256;
/// Workgroups computing partial sums for the acceleration step.
const PARTIALS: u32 = 256;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Allow or forbid the GPU (the CPU is used when forbidden).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// The GPU, if allowed and available (created on first use; that can take a moment, so call
/// it once from a background thread at start-up).
pub fn get() -> Option<&'static Gpu> {
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    probe()
}

/// Look for the GPU (once), whether or not it is allowed.
pub fn probe() -> Option<&'static Gpu> {
    GPU.get_or_init(|| if std::env::var_os("PS_NO_GPU").is_some() { None } else { Gpu::new() }).as_ref()
}

/// Below this many pixels the CPU is quicker than a round trip to the GPU.
const MIN_PIXELS: usize = 1 << 18;

/// How much computing an operation does per pixel moved to and from the GPU.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Work {
    /// Many operations per pixel (bilateral filter): worth it on any GPU.
    Heavy,
    /// A few operations per pixel (blurs, starlet scales, min/max). On a separate graphics
    /// card copying the plane over the bus costs about as much as the CPU takes, so only
    /// GPUs sharing memory with the CPU (Apple Silicon, integrated graphics) are used.
    Light,
}

/// The GPU, if allowed and available, for a `w`×`h` plane worth sending there.
pub fn for_plane(w: usize, h: usize, work: Work) -> Option<&'static Gpu> {
    get().filter(|g| w * h >= MIN_PIXELS && g.fits(w, h) && (work == Work::Heavy || g.shared_memory))
}

/// What was found, without waiting: `None` while still looking, `Some(None)` if there is no
/// usable GPU.
pub fn status() -> Option<Option<&'static Gpu>> {
    GPU.get().map(Option::as_ref)
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Adapter name and API, for display.
    pub name: String,
    /// Integrated GPU using the computer's memory: moving data to it is cheap.
    pub shared_memory: bool,
    max_buffer: u64,
    layout: wgpu::BindGroupLayout,
    kernels: Kernels,
    ops: ops::Ops,
}

struct Kernels {
    start: wgpu::ComputePipeline,
    conv_rows: wgpu::ComputePipeline,
    conv_cols: wgpu::ComputePipeline,
    extrapolate: wgpu::ComputePipeline,
    ratio: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    difference: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    step_length: wgpu::ComputePipeline,
    residual: wgpu::ComputePipeline,
    hold: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    radius: u32,
    stride: u32,
    tv_lambda: f32,
    damp: f32,
    partials: u32,
    signal: f32,
}

impl Gpu {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..wgpu::InstanceDescriptor::new_without_display_handle() });
        let options = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() };
        let adapter = pollster::block_on(instance.request_adapter(&options)).ok()?;
        let info = adapter.get_info();
        // A software rasteriser would be slower than our CPU code.
        if info.device_type == wgpu::DeviceType::Cpu {
            return None;
        }
        let limits = adapter.limits();
        let desc = wgpu::DeviceDescriptor { label: Some("planetary-studio compute"), required_limits: limits.clone(), ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;

        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deconv"),
            entries: &[uniform, storage(1, true), storage(2, true), storage(3, false), storage(4, true), storage(5, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("deconv"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("deconv"), source: wgpu::ShaderSource::Wgsl(include_str!("deconv.wgsl").into()) });
        let kernel = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let kernels = Kernels {
            start: kernel("start"),
            conv_rows: kernel("conv_rows"),
            conv_cols: kernel("conv_cols"),
            extrapolate: kernel("extrapolate"),
            ratio: kernel("ratio"),
            update: kernel("update"),
            difference: kernel("difference"),
            reduce: kernel("reduce"),
            step_length: kernel("step_length"),
            residual: kernel("residual"),
            hold: kernel("hold"),
            add: kernel("add"),
        };
        let name = format!("{} ({:?})", info.name, info.backend);
        let max_buffer = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let ops = ops::Ops::new(&device);
        let shared_memory = info.device_type == wgpu::DeviceType::IntegratedGpu;
        Some(Self { device, queue, name, shared_memory, max_buffer, layout, kernels, ops })
    }

    /// Whether a `w`×`h` plane fits in one GPU buffer.
    pub fn fits(&self, w: usize, h: usize) -> bool {
        (w * h * 4) as u64 <= self.max_buffer
    }

    /// Richardson–Lucy with a PSF that is a sum of separable `terms`, on an already padded
    /// `w`×`h` plane `d`. Same algorithm, options and safeguards as the CPU version, which it
    /// reports the iterations of. `Ok(None)` means cancelled; `Err` means the GPU could not do
    /// it (the caller falls back to the CPU).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn richardson_lucy(
        &self,
        d: &[f32],
        w: usize,
        h: usize,
        terms: &[crate::deconv::SeparableTerm],
        p: &crate::deconv::DeconvParams,
        damp: f32,
        guards: &crate::deconv::Guards,
        cancel: &dyn Fn() -> bool,
        progress: &dyn Fn(f32),
    ) -> Result<Option<(Vec<f32>, u32)>, String> {
        let (iterations, accelerate, tv_lambda) = (p.iterations, p.accelerate, p.tv_lambda);
        // The anti-ringing limits are three planes in one buffer.
        if !self.fits(w, h) || (guards.clamp.is_some() && !self.fits(3 * w, h)) {
            return Err("immagine troppo grande per la memoria della GPU".into());
        }
        let n = w * h;
        let groups = (n as u32).div_ceil(GROUP);
        let groups_x = groups.min(32768);
        let groups_y = groups.div_ceil(groups_x);
        let params = Params { w: w as u32, h: h as u32, radius: 0, stride: groups_x * GROUP, tv_lambda, damp, partials: PARTIALS, signal: guards.stop.map_or(f32::MAX, |s| s.0) };

        let dev = &self.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let init = |label: &str, contents: &[u8], usage| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
        let plane = |label: &str| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniforms = init("params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);
        // Per term: its parameters (the radius differs) and row and column taps.
        let term_buffers: Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> = terms
            .iter()
            .map(|(rows, cols)| {
                let term = Params { radius: (rows.len() / 2) as u32, ..params };
                (
                    init("term", bytemuck::bytes_of(&term), wgpu::BufferUsages::UNIFORM),
                    init("rows", bytemuck::cast_slice(rows), wgpu::BufferUsages::STORAGE),
                    init("cols", bytemuck::cast_slice(cols), wgpu::BufferUsages::STORAGE),
                )
            })
            .collect();
        let part = (terms.len() > 1).then(|| plane("part"));
        let scalars_size = (4 * (1 + 2 * PARTIALS)) as u64;
        let scalars = init("scalars", bytemuck::cast_slice(&vec![0.0f32; 1 + 2 * PARTIALS as usize]), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        let scalars_back = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalars readback"),
            size: scalars_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let limits = guards.clamp.as_ref().map(|c| {
            let interleaved: Vec<f32> = (0..n).flat_map(|i| [c.lo[i], c.hi[i], c.weight[i]]).collect();
            init("limits", bytemuck::cast_slice(&interleaved), wgpu::BufferUsages::STORAGE)
        });
        // The observed plane is the only one sent over; both starting estimates are made from it.
        let observed = init("observed", bytemuck::cast_slice(d), wgpu::BufferUsages::STORAGE);
        let (mut x, mut x_prev) = (plane("x"), plane("x_prev"));
        let (y, tmp, model, ratio, correction) = (plane("y"), plane("tmp"), plane("model"), plane("ratio"), plane("correction"));
        let (mut g1, mut g2) = if accelerate { (plane("g1"), plane("g2")) } else { (plane("g"), plane("g")) };
        let staging =
            dev.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (n * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("memoria GPU insufficiente: {e}"));
        }

        let bind_with = |params: &wgpu::Buffer, taps: &wgpu::Buffer, a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: taps.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: scalars.as_entire_binding() },
                ],
            })
        };
        // Unused slots of a kernel's bind group get any buffer of the right kind.
        let bind = |a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer| bind_with(&uniforms, &term_buffers[0].1, a, b, out);
        let tiles = ((w as u32).div_ceil(16), (h as u32).div_ceil(16));
        let run = |pass: &mut wgpu::ComputePass, kernel: &wgpu::ComputePipeline, group: &wgpu::BindGroup, grid: (u32, u32)| {
            pass.set_pipeline(kernel);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        };
        let k = &self.kernels;
        let elements = (groups_x, groups_y);
        // dst = src blurred by the PSF (symmetric, so also its correlation): term by term.
        let blur = |pass: &mut wgpu::ComputePass, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            for (t, (params, rows, cols)) in term_buffers.iter().enumerate() {
                run(pass, &k.conv_rows, &bind_with(params, rows, src, src, &tmp), tiles);
                let target = if t == 0 { dst } else { part.as_ref().expect("several terms") };
                run(pass, &k.conv_cols, &bind_with(params, cols, &tmp, &tmp, target), tiles);
                if t > 0 {
                    run(pass, &k.add, &bind(target, target, dst), elements);
                }
            }
        };
        let mut corrections = 0;
        let mut encoder = dev.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            run(&mut pass, &k.start, &bind(&observed, &observed, &x), elements);
            run(&mut pass, &k.start, &bind(&observed, &observed, &x_prev), elements);
        }
        self.queue.submit([encoder.finish()]);

        let mut done = iterations.max(1);
        for iteration in 0..iterations.max(1) {
            if cancel() {
                return Ok(None);
            }
            let accelerating = accelerate && corrections >= 2;
            // Stopping rule (as on the CPU): fitted down to the noise, measured on the model of
            // this iteration's starting point.
            let measuring = guards.stop.is_some() && iteration > 0;
            if !accelerating {
                self.queue.write_buffer(&scalars, 0, bytemuck::bytes_of(&0.0f32));
            }
            let mut encoder = dev.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                if accelerating {
                    run(&mut pass, &k.reduce, &bind(&g1, &g2, &tmp), (PARTIALS, 1));
                    run(&mut pass, &k.step_length, &bind(&g1, &g2, &tmp), (1, 1));
                }
                run(&mut pass, &k.extrapolate, &bind(&x, &x_prev, &y), elements);
                blur(&mut pass, &y, &model);
                if measuring {
                    // How well the current estimate explains the data, read back below.
                    run(&mut pass, &k.residual, &bind(&observed, &model, &tmp), (PARTIALS, 1));
                }
                run(&mut pass, &k.ratio, &bind(&observed, &model, &ratio), elements);
                // The PSF is symmetric: correlating with it is convolving with it.
                blur(&mut pass, &ratio, &correction);
                // The new estimate goes where x_prev was; x becomes x_prev below.
                run(&mut pass, &k.update, &bind(&y, &correction, &x_prev), elements);
                if let Some(limits) = &limits {
                    run(&mut pass, &k.hold, &bind(limits, limits, &x_prev), elements);
                }
                if accelerate {
                    run(&mut pass, &k.difference, &bind(&x_prev, &y, &g2), elements);
                }
            }
            if measuring {
                encoder.copy_buffer_to_buffer(&scalars, 0, &scalars_back, 0, scalars_size);
            }
            self.queue.submit([encoder.finish()]);
            // One iteration at a time, so cancelling and progress stay responsive.
            dev.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
            if let (true, Some((_, target))) = (measuring, guards.stop) {
                let sums = read_back(dev, &scalars_back)?;
                let (sum, count) = (0..PARTIALS as usize).fold((0.0f64, 0.0f64), |(s, c), j| (s + sums[1 + 2 * j] as f64, c + sums[2 + 2 * j] as f64));
                if count > 0.0 && sum / count <= target {
                    // Keep the estimate this iteration started from (the CPU stops there too).
                    done = iteration;
                    break;
                }
            }
            if accelerate {
                std::mem::swap(&mut g1, &mut g2);
                corrections += 1;
            }
            std::mem::swap(&mut x, &mut x_prev);
            progress((iteration + 1) as f32 / iterations.max(1) as f32);
        }

        let mut encoder = dev.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&x, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit([encoder.finish()]);
        progress(1.0);
        Ok(Some((read_back(dev, &staging)?, done)))
    }
}

/// Wait for the GPU and read a mappable buffer.
fn read_back(dev: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<f32>, String> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    dev.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    let view = slice.get_mapped_range().map_err(|e| e.to_string())?;
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    buffer.unmap();
    Ok(out)
}
