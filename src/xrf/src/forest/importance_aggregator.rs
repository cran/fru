pub struct ImportanceAggregator {
    n: usize,
    mean: f64,
    sum_sq: f64,
}

impl Default for ImportanceAggregator {
    fn default() -> Self {
        ImportanceAggregator::new()
    }
}

impl ImportanceAggregator {
    pub fn new() -> Self {
        ImportanceAggregator {
            n: 0,
            mean: 0.,
            sum_sq: 0.,
        }
    }
    pub fn ingest(&mut self, x: f64) {
        self.n += 1;
        let old_mean = self.mean;
        self.mean += (x - old_mean) / (self.n as f64);
        self.sum_sq += (x - old_mean) * (x - self.mean);
    }
    pub fn merge(&mut self, other: &Self) {
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        self.mean = ((self.n as f64) * self.mean + (other.n as f64) * other.mean) / (n as f64);
        self.sum_sq +=
            other.sum_sq + delta * delta * ((self.n as f64) * (other.n as f64)) / (n as f64);
        self.n += other.n;
    }
    pub fn value(&self, count: usize) -> f64 {
        assert!(count >= self.n);
        (self.n as f64) / (count as f64) * self.mean
    }
    pub fn value_normalised(&self, count: usize) -> Option<f64> {
        assert!(count >= self.n);
        let on = count - self.n;
        if self.n <= 2 {
            None
        } else {
            let sum_sq = self.sum_sq
                + self.mean * self.mean * ((self.n as f64) * (on as f64)) / (count as f64);
            let mean = (self.n as f64) / (count as f64) * self.mean;
            let sd = (sum_sq / ((count - 1) as f64)).sqrt();
            if sd == 0. { None } else { Some(mean / sd) }
        }
    }
    pub fn from_raw(raw: &[u8; 24]) -> Self {
        Self {
            n: usize::from_le_bytes(raw[0..8].try_into().unwrap()),
            mean: f64::from_le_bytes(raw[8..16].try_into().unwrap()),
            sum_sq: f64::from_le_bytes(raw[16..24].try_into().unwrap()),
        }
    }
    pub fn into_raw(&self) -> [u8; 24] {
        [
            self.n.to_le_bytes(),
            self.mean.to_le_bytes(),
            self.sum_sq.to_le_bytes(),
        ]
        .concat()
        .try_into()
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let p: Vec<_> = (1..=48).map(|x| x as f64).collect();
        let mut ia = ImportanceAggregator::new();
        p.iter().for_each(|&x| ia.ingest(x));
        assert_eq!(ia.value(48), 24.5);
        assert_eq!(ia.value_normalised(48).unwrap(), 1.75);
    }
    #[test]
    fn merges() {
        let mut ia1 = ImportanceAggregator::new();
        let mut ia2 = ImportanceAggregator::new();
        (1..27).for_each(|x| ia1.ingest(x as f64));
        (27..=48).for_each(|x| ia2.ingest(x as f64));
        ia1.merge(&ia2);
        assert_eq!(ia1.value(48), 24.5);
        assert_eq!(ia1.value_normalised(48).unwrap(), 1.75);
    }
    #[test]
    fn add_zeroes() {
        let mut ia1 = ImportanceAggregator::new();
        let mut ia2 = ImportanceAggregator::new();
        let c1 = 48;
        let c2 = 12;
        (1..c1)
            .chain(std::iter::repeat_n(0, c2))
            .for_each(|x| ia1.ingest(x as f64));
        (1..c1).for_each(|x| ia2.ingest(x as f64));
        assert_eq!(ia1.value(c1 + c2), ia2.value(c1 + c2));
        assert_eq!(ia1.value_normalised(c1 + c2), ia2.value_normalised(c1 + c2));
    }
}
