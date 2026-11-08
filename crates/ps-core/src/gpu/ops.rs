//! GPU versions of the filters most of the processing is built from. Each takes and returns
//! planes in main memory (one upload, one download per call) and mirrors a CPU function, which
//! the callers fall back to on any error.

use super::Gpu;
use wgpu::util::DeviceExt;

const GROUP: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    r: u32,
    step: u32,
    stride: u32,
    mode: u32,
    a: f32,
    b: f32,
}

pub(super) struct Ops {
    layout: wgpu::BindGroupLayout,
    conv_rows: wgpu::ComputePipeline,
    conv_cols: wgpu::ComputePipeline,
    subtract: wgpu::ComputePipeline,
    box_rows: wgpu::ComputePipeline,
    box_cols: wgpu::ComputePipeline,
    bilateral: wgpu::ComputePipeline,
    extreme_rows: wgpu::ComputePipeline,
    extreme_cols: wgpu::ComputePipeline,
}

impl Ops {
    pub(super) fn new(device: &wgpu::Device) -> Self {
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
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ops"), entries: &[uniform, storage(1, true), storage(2, true), storage(3, false), storage(4, true)] });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ops"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ops"), source: wgpu::ShaderSource::Wgsl(include_str!("ops.wgsl").into()) });
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
        Self {
            conv_rows: kernel("conv_rows"),
            conv_cols: kernel("conv_cols"),
            subtract: kernel("subtract"),
            box_rows: kernel("box_rows"),
            box_cols: kernel("box_cols"),
            bilateral: kernel("bilateral"),
            extreme_rows: kernel("extreme_rows"),
            extreme_cols: kernel("extreme_cols"),
            layout,
        }
    }
}

/// Buffers and recorded work for one `w`×`h` job.
struct Job<'g> {
    gpu: &'g Gpu,
    w: usize,
    h: usize,
    encoder: wgpu::CommandEncoder,
    /// Placeholder for unused bindings.
    dummy: wgpu::Buffer,
}

impl<'g> Job<'g> {
    fn new(gpu: &'g Gpu, w: usize, h: usize) -> Result<Self, String> {
        if !gpu.fits(w, h) {
            return Err("immagine troppo grande per la memoria della GPU".into());
        }
        let encoder = gpu.device.create_command_encoder(&Default::default());
        let dummy = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("unused"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        Ok(Self { gpu, w, h, encoder, dummy })
    }

    fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("plane"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn plane(&self) -> wgpu::Buffer {
        self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plane"),
            size: (self.w * self.h * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn taps(&self, taps: &[f32]) -> wgpu::Buffer {
        self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("taps"), contents: bytemuck::cast_slice(taps), usage: wgpu::BufferUsages::STORAGE })
    }

    /// Record one kernel over the image (`tiled`: 16×16 groups over x, y; else element-wise).
    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, kernel: &wgpu::ComputePipeline, tiled: bool, mut params: Params, a: &wgpu::Buffer, b: Option<&wgpu::Buffer>, out: &wgpu::Buffer, taps: Option<&wgpu::Buffer>) {
        let (w, h) = (self.w as u32, self.h as u32);
        let groups = (w * h).div_ceil(GROUP);
        let groups_x = groups.min(32768);
        (params.w, params.h, params.stride) = (w, h, groups_x * GROUP);
        let device = &self.gpu.device;
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.gpu.ops.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.unwrap_or(&self.dummy).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: taps.unwrap_or(&self.dummy).as_entire_binding() },
            ],
        });
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(kernel);
        pass.set_bind_group(0, &group, &[]);
        if tiled {
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        } else {
            pass.dispatch_workgroups(groups_x, groups.div_ceil(groups_x), 1);
        }
    }

    /// Run what was recorded and read `buffers` back, in order.
    fn finish(self, buffers: &[&wgpu::Buffer]) -> Result<Vec<Vec<f32>>, String> {
        let Job { gpu, w, h, mut encoder, .. } = self;
        let size = (w * h * 4) as u64;
        let staging: Vec<wgpu::Buffer> = buffers
            .iter()
            .map(|b| {
                let s =
                    gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                encoder.copy_buffer_to_buffer(b, 0, &s, 0, size);
                s
            })
            .collect();
        gpu.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        for (k, s) in staging.iter().enumerate() {
            let tx = tx.clone();
            s.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send((k, r)).ok();
            });
        }
        drop(tx);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        for (_, r) in rx.iter() {
            r.map_err(|e| e.to_string())?;
        }
        staging
            .iter()
            .map(|s| {
                let view = s.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
                let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
                drop(view);
                s.unmap();
                Ok(out)
            })
            .collect()
    }
}

impl Gpu {
    /// [`crate::mask::gaussian_approx`]: three box blurs of radius `r` along rows and columns.
    pub fn box_blur3(&self, plane: &[f32], w: usize, h: usize, r: usize) -> Result<Vec<f32>, String> {
        let mut job = Job::new(self, w, h)?;
        let (a, b) = (job.upload(plane), job.plane());
        let p = Params { r: r as u32, ..Default::default() };
        for _ in 0..3 {
            job.run(&self.ops.box_rows, true, p, &a, None, &b, None);
            job.run(&self.ops.box_cols, true, p, &b, None, &a, None);
        }
        Ok(job.finish(&[&a])?.swap_remove(0))
    }

    /// Starlet (à trous) transform: smoothing with `taps` spaced `steps[j]` apart, level after
    /// level. Returns the detail layers (empty unless `details`) and the final smooth plane.
    pub fn starlet(&self, plane: &[f32], w: usize, h: usize, taps: &[f32], steps: &[usize], details: bool) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let passes: Vec<(&[f32], usize)> = steps.iter().map(|&s| (taps, s)).collect();
        self.cascade(plane, w, h, &passes, details)
    }

    /// Like [`Self::starlet`] with its own taps and spacing for each pass.
    pub fn cascade(&self, plane: &[f32], w: usize, h: usize, passes: &[(&[f32], usize)], details: bool) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let mut job = Job::new(self, w, h)?;
        let (mut current, mut next, tmp) = (job.upload(plane), job.plane(), job.plane());
        let layers: Vec<wgpu::Buffer> = if details { passes.iter().map(|_| job.plane()).collect() } else { Vec::new() };
        for (j, &(taps, step)) in passes.iter().enumerate() {
            let taps_buf = job.taps(taps);
            let p = Params { r: (taps.len() / 2) as u32, step: step as u32, ..Default::default() };
            job.run(&self.ops.conv_rows, true, p, &current, None, &tmp, Some(&taps_buf));
            job.run(&self.ops.conv_cols, true, p, &tmp, None, &next, Some(&taps_buf));
            if details {
                job.run(&self.ops.subtract, false, p, &current, Some(&next), &layers[j], None);
            }
            std::mem::swap(&mut current, &mut next);
        }
        let mut wanted: Vec<&wgpu::Buffer> = layers.iter().collect();
        wanted.push(&current);
        let mut out = job.finish(&wanted)?;
        let residual = out.pop().expect("residual requested");
        Ok((out, residual))
    }

    /// [`crate::enhance`]'s bilateral filter.
    pub fn bilateral(&self, plane: &[f32], w: usize, h: usize, sigma_s: f32, sigma_r: f32) -> Result<Vec<f32>, String> {
        let mut job = Job::new(self, w, h)?;
        let (a, b) = (job.upload(plane), job.plane());
        let p = Params { r: (2.0 * sigma_s).ceil() as u32, a: sigma_s, b: -1.0 / (2.0 * sigma_r * sigma_r), ..Default::default() };
        job.run(&self.ops.bilateral, true, p, &a, None, &b, None);
        Ok(job.finish(&[&b])?.swap_remove(0))
    }

    /// Minimum and maximum over the (2r+1)² square around each pixel.
    pub fn min_max(&self, plane: &[f32], w: usize, h: usize, r: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut job = Job::new(self, w, h)?;
        let (src, tmp, lo, hi) = (job.upload(plane), job.plane(), job.plane(), job.plane());
        for (mode, out) in [(0, &lo), (1, &hi)] {
            let p = Params { r: r as u32, mode, ..Default::default() };
            job.run(&self.ops.extreme_rows, true, p, &src, None, &tmp, None);
            job.run(&self.ops.extreme_cols, true, p, &tmp, None, out, None);
        }
        let mut out = job.finish(&[&lo, &hi])?;
        let hi = out.pop().expect("two planes");
        Ok((out.pop().expect("two planes"), hi))
    }
}

#[cfg(test)]
mod tests {
    use crate::conv::convolve_separable;

    /// A test plane with edges, gradients and fine texture, and a size that is not a multiple
    /// of the 16×16 tiles.
    fn plane(w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let disk = if (x - 90.0).hypot(y - 60.0) < 40.0 { 0.5 } else { 0.05 };
                disk + 0.1 * (x * 0.37).sin() * (y * 0.23).cos() + ((i * 7919) % 97) as f32 * 1e-3
            })
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    const W: usize = 203;
    const H: usize = 131;

    #[test]
    fn gpu_filters_match_cpu() {
        let Some(gpu) = crate::gpu::probe() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let p = plane(W, H);

        for r in [1, 4, 40] {
            let cpu = crate::mask::box_blur3(&p, W, H, r);
            let err = max_diff(&gpu.box_blur3(&p, W, H, r).unwrap(), &cpu);
            assert!(err < 1e-5, "box blur r {r}: {err}");
        }

        let taps = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        let steps = [1, 2, 4, 8, 16];
        let (layers, residual) = gpu.starlet(&p, W, H, &taps, &steps, true).unwrap();
        let (mut current, mut next, mut tmp) = (p.clone(), vec![0.0; W * H], vec![0.0; W * H]);
        for (j, &step) in steps.iter().enumerate() {
            convolve_separable(&current, &mut next, &mut tmp, W, H, &taps, step);
            let detail: Vec<f32> = current.iter().zip(&next).map(|(c, n)| c - n).collect();
            assert!(max_diff(&layers[j], &detail) < 1e-5, "starlet layer {j}");
            std::mem::swap(&mut current, &mut next);
        }
        assert!(max_diff(&residual, &current) < 1e-5, "starlet residual");

        let err = max_diff(&gpu.bilateral(&p, W, H, 1.5, 0.05).unwrap(), &crate::enhance::bilateral_cpu(&p, W, H, 1.5, 0.05));
        assert!(err < 1e-4, "bilateral: {err}");

        for r in [1, 3] {
            let (lo, hi) = gpu.min_max(&p, W, H, r).unwrap();
            let (clo, chi) = crate::dering::min_max_cpu(&p, W, H, r);
            assert_eq!((lo, hi), (clo, chi), "min/max r {r}");
        }
    }
}
