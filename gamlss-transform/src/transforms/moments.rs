//! Streaming population moments shared by fitted target transforms.

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Moments {
    count: f64,
    mean: f64,
    sum_squares: f64,
}

impl Moments {
    pub(super) fn push(&mut self, value: f64) {
        self.count += 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
        self.sum_squares = delta.mul_add(value - self.mean, self.sum_squares);
    }

    pub(super) fn push_finite(&mut self, value: f64) -> Option<()> {
        if !value.is_finite() {
            return None;
        }
        self.push(value);
        Some(())
    }

    pub(super) const fn count(self) -> f64 {
        self.count
    }

    pub(super) const fn mean(self) -> f64 {
        self.mean
    }

    pub(super) fn variance(self) -> f64 {
        self.sum_squares / self.count
    }
}
