pub fn random(seed: &mut u64, lo: f64, hi: f64) -> f64 {
    if lo >= hi {
        return lo;
    }
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 32) as f64 / (u32::MAX as f64)) * (hi - lo) + lo
}
