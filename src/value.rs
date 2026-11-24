use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;
use std::vec;

pub struct ValueInner {
    pub data: f64,
    pub grad: f64,
    pub prevs: Vec<Value>,
    pub op: String,
    pub backward: Box<dyn Fn()>,
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            grad: 0f64,
            prevs: Vec::with_capacity(8),
            op: String::with_capacity(8),
            backward: Box::new(|| {}),
        }
    }
}

#[derive(Clone)]
pub struct Value {
    pub inner: Rc<RefCell<ValueInner>>,
}

impl Value {
    pub fn new(data: f64) -> Self {
        let val_inner = ValueInner::new(data);
        Self {
            inner: Rc::new(RefCell::new(val_inner)),
        }
    }

    pub fn with_op(data: f64, op: String, lhs: Value, rhs: Value) -> Self {
        let prevs = vec![lhs, rhs];
        let val_inner = ValueInner {
            data,
            grad: 0f64,
            prevs,
            op,
            backward: Box::new(|| {}),
        };

        Self {
            inner: Rc::new(RefCell::new(val_inner)),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    pub fn backward(&mut self) {
        let mut visited: HashSet<usize> = HashSet::with_capacity(16);
        let mut sorted_values: Vec<Value> = Vec::with_capacity(16);

        Self::topo_sort(self.clone(), &mut visited, &mut sorted_values);

        self.inner.borrow_mut().grad = 1f64;
        for val in sorted_values.into_iter().rev() {
            // 对inner的借用会持续到闭包调用完成，闭包如果会修改当前节点自己（目前不会）则需要用Rc
            // clone出闭包后结束借用再调用
            (val.inner.borrow().backward)();
        }
    }

    fn topo_sort(value: Value, visited: &mut HashSet<usize>, results: &mut Vec<Value>) {
        let id = value.inner.as_ptr() as usize;
        if !visited.contains(&id) {
            visited.insert(id);
            for child in value.inner.borrow().prevs.iter() {
                Self::topo_sort(child.clone(), visited, results);
            }

            results.push(value);
        }
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        let self_data = self.inner.borrow().data;
        let other_data = other.inner.borrow().data;

        let out_val = Value::with_op(
            self_data + other_data,
            "+".into(),
            self.clone(),
            other.clone(),
        );

        let out_weak = Rc::downgrade(&out_val.inner);
        out_val.inner.borrow_mut().backward = Box::new(move || {
            if let Some(up_value) = out_weak.upgrade() {
                let out_grad = up_value.borrow().grad;
                self.inner.borrow_mut().grad += out_grad;
                other.inner.borrow_mut().grad += out_grad;
            }
        });

        out_val
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let self_data = self.inner.borrow().data;
        let other_data = other.inner.borrow().data;
        let out_val = Value::with_op(
            self_data * other_data,
            "*".into(),
            self.clone(),
            other.clone(),
        );

        let out_weak = Rc::downgrade(&out_val.inner);
        out_val.inner.borrow_mut().backward = Box::new(move || {
            if let Some(up_value) = out_weak.upgrade() {
                let out_grad = up_value.borrow().grad;
                self.inner.borrow_mut().grad += out_grad * other_data;
                other.inner.borrow_mut().grad += out_grad * self_data;
            }
        });

        out_val
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * Value::new(-1.0)
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

// 为方便与 f64 进行混合运算
impl Add<f64> for Value {
    type Output = Value;
    fn add(self, other: f64) -> Value {
        self + Value::new(other)
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, other: f64) -> Value {
        self * Value::new(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        // L = (a * b + c) * f + a = (2 * 3 + 10) * 2 + 2 = 34
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::new(10.0);
        let d = a.clone() * b.clone();
        let e = d + c.clone();
        let f = Value::new(2.0);
        let g = e * f.clone();
        let mut l = g + a.clone();
        assert_eq!(l.data(), 34.0);

        // dL/dl = 1
        // dL/dg = 1
        // dL/da = 1  // first
        // dL/df = dL/dg * e = 16
        // dL/de = dL/dg * f = 2
        // dL/dc = dL/de * 1 = 2
        // dL/dd = dL/de * 1 = 2
        // dL/da += dL/dd * b = 1 + 2 * 3 = 7 // second
        // dL/db = dL/dd * a = 2 * 2 = 4

        l.backward();
        assert_eq!(a.grad(), 7.0);
        assert_eq!(b.grad(), 4.0);
        assert_eq!(c.grad(), 2.0);
        assert_eq!(f.grad(), 16.0);

        let a = Value::new(2.0);
        let mut b = a.clone() + a.clone();
        b.backward();
        assert_eq!(a.grad(), 2.0);

        let a = Value::new(2.0);
        let mut b = a.clone() * a.clone();
        b.backward();
        assert_eq!(a.grad(), 4.0);
    }
}
