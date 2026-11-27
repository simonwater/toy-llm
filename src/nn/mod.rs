mod mlp;

use crate::{Value, utils};
pub use mlp::{Layer, MLP};

pub trait Module {
    fn parameters(&self) -> Vec<Value>;
    fn zero_grad(&self) {
        for p in &self.parameters() {
            p.set_grad(0.0);
        }
    }
}

/// 神经元节点
pub struct Neuron {
    pub weights: Vec<Value>, // 权重
    pub b: Value,            // 偏置
    pub nonline: bool,       // 非线性激活函数
    pub param_size: usize,   // 参数个数
}

impl Neuron {
    pub fn new(nin: usize, nonline: bool, seed: &mut u64) -> Self {
        // LCG 生成 [-1, 1] 的随机初始权重
        let mut weights = Vec::with_capacity(nin);
        for _ in 0..nin {
            let val = utils::random(seed, -1.0, 1.0);
            weights.push(Value::new(val));
        }
        let b = Value::new(0.0);
        Self {
            weights,
            b,
            nonline,
            param_size: nin + 1,
        }
    }

    /// 前向推理
    pub fn forward(&self, inputs: &[Value]) -> Value {
        let mut act = self.b.clone();
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            act = act + w.clone() * x.clone();
        }

        if self.nonline { act.tanh() } else { act }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.b.clone());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_test() {
        let n = Neuron::new(2, false, &mut 40);
        let x = vec![Value::new(2.0), Value::new(3.0)];
        let y = n.forward(&x);
        println!("{}", y.data());
    }
}
