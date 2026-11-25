mod mlp;

use crate::Value;
pub use mlp::{Layer, MLP};

/// 神经元节点
pub struct Neuron {
    pub weights: Vec<Value>, // 权重
    pub b: Value,            // 偏置
    pub nonline: bool,       // 非线性激活函数
}

impl Neuron {
    pub fn new(nin: usize, nonline: bool, seed: &mut u64) -> Self {
        // LCG 生成 [-1, 1] 的随机初始权重
        let mut weights = Vec::with_capacity(nin);
        for _ in 0..nin {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((*seed >> 32) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
            weights.push(Value::new(val));
        }
        let b = Value::new(0.0);
        Self {
            weights,
            b,
            nonline,
        }
    }

    /// 前向推理
    pub fn forward(&self, inputs: &[Value]) -> Value {
        let mut act = self.b.clone();
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            act = act + w.clone() * x.clone();
        }

        if self.nonline { act.relu() } else { act }
    }
}
