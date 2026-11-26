use super::{Module, Neuron};
use crate::Value;

/// 全连接层
pub struct Layer {
    neurons: Vec<Neuron>,
    param_size: usize,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonline: bool, seed: &mut u64) -> Self {
        let mut neurons = Vec::with_capacity(nout);
        let mut param_size = 0;
        for _ in 0..nout {
            let n = Neuron::new(nin, nonline, seed);
            param_size += n.param_size;
            neurons.push(n);
        }
        Self {
            neurons,
            param_size,
        }
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

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::with_capacity(self.param_size);
        for n in &self.neurons {
            params.extend(n.parameters());
        }
        params
    }
}

/// 多层感知机
pub struct MLP {
    layers: Vec<Layer>,
    param_size: usize,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize], seed: &mut u64) -> Self {
        let mut layers = Vec::with_capacity(nouts.len());
        let mut param_size = 0;
        let mut in_outs = vec![nin];
        in_outs.extend_from_slice(nouts);
        for i in 0..nouts.len() {
            let nonline = i == nouts.len() - 1; //除最后一层，其余用非线性激活函数
            let layer = Layer::new(in_outs[i], in_outs[i + 1], nonline, seed);
            param_size += layer.param_size;
            layers.push(layer);
        }
        Self { layers, param_size }
    }

    /// 前向推理
    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut cur = inputs.to_vec();
        for layer in self.layers.iter() {
            cur = layer.forward(&cur);
        }

        cur
    }

    pub fn forward_one(&self, inputs: &[Value]) -> Value {
        let out = self.forward(inputs);
        assert_eq!(out.len(), 1);
        out[0].clone()
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::with_capacity(self.param_size);
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
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
        println!("outputs: {:?}", outs);

        // 重置所有参数梯度
        mlp.zero_grad();
        let params = mlp.parameters();
        let zero_grad_cnt = params.iter().filter(|p| p.grad() == 0.0).count();
        assert_eq!(zero_grad_cnt, mlp.param_size);

        // 反向传播计算所有参数的梯度
        out.backward();
        let zero_grad_cnt = params.iter().filter(|p| p.grad() == 0.0).count();
        assert_eq!(zero_grad_cnt, 0);
    }

    #[test]
    fn test_multi_inputs() {
        let xs = vec![
            vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
            vec![Value::new(3.0), Value::new(1.0), Value::new(0.5)],
            vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
            vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
        ];
        // 真实值
        let ys = vec![
            Value::new(1.0),
            Value::new(-1.0),
            Value::new(-1.0),
            Value::new(1.0),
        ];

        let mlp = MLP::new(3, &[4, 4, 1], &mut 42);
        let parameters = mlp.parameters();

        for i in 1..=20 {
            println!("first p: {:?}", parameters[0]);
            // 预测值
            let mut ypreds = Vec::<Value>::with_capacity(xs.len());
            for x in &xs {
                let ypred = mlp.forward_one(x);
                ypreds.push(ypred);
            }
            // 损失值
            let mut loss = Value::new(0.0);
            for (y, ypred) in ys.iter().zip(ypreds.iter()) {
                loss = loss + (y.clone() - ypred.clone()).pow(2.0);
            }
            println!("{}.loss: {}", i, loss);

            // 反向传播计算梯度
            mlp.zero_grad();
            loss.backward();

            // 梯度法更新参数
            for p in parameters.iter() {
                let d = p.data() - 0.01 * p.grad();
                p.set_data(d);
            }
            println!()
        }

        let ypred: Vec<Value> = xs.iter().map(|x| mlp.forward_one(x)).collect();
        println!();
        println!("pred: {:?}", ypred);
    }
}
