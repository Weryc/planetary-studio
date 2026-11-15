//! A small seeded generator: every frame is reproducible from (seed, frame index).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03 | 1);
        for _ in 0..8 {
            r.next();
        }
        r
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn gauss(&mut self) -> f32 {
        ((-2.0 * self.uniform().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()) as f32
    }

    /// Poisson sample of mean `m` (normal approximation above 40).
    pub fn poisson(&mut self, m: f32) -> f32 {
        if m <= 0.0 {
            return 0.0;
        }
        if m > 40.0 {
            return (m + m.sqrt() * self.gauss()).round().max(0.0);
        }
        let limit = (-m as f64).exp();
        let (mut k, mut p) = (0.0f32, self.uniform());
        while p > limit {
            k += 1.0;
            p *= self.uniform();
        }
        k
    }
}
