use super::Neuron;
use crate::Value;

/// 全连接层
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonline: bool, seed: &mut u64) -> Self {
        let mut neurons = Vec::with_capacity(nout);
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, nonline, seed));
        }
        Self { neurons }
    }

    /// 前向推理
    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut vals = Vec::with_capacity(self.neurons.len());
        for n in self.neurons.iter() {
            let val = n.forward(inputs);
            vals.push(val);
        }

        vals
    }
}

/// 多层感知机
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize], seed: &mut u64) -> Self {
        let mut layers = Vec::with_capacity(nouts.len());
        let mut in_outs = vec![nin];
        in_outs.extend_from_slice(nouts);
        for i in 0..nouts.len() {
            let nonline = i == nouts.len() - 1; //除最后一层，其余用非线性激活函数
            let layer = Layer::new(in_outs[i], in_outs[i + 1], nonline, seed);
            layers.push(layer);
        }
        Self { layers }
    }

    /// 前向推理
    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut cur = inputs.to_vec();
        for layer in self.layers.iter() {
            cur = layer.forward(&cur);
        }

        cur
    }

    pub fn zero_grad(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_forward_and_backward() {
        let mut seed = 42;
        // 创建一个 3 输入，输出层结构为 [4, 4, 1] 的多层感知机
        let mlp = MLP::new(3, &[4, 4, 1], &mut seed);

        let x = vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)];
        let outs = mlp.forward(&x);
        let out = outs.first().expect("感知机结构不合法");
        println!("out value: {}", out.data());
    }
}
