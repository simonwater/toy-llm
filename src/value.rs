use std::cell::RefCell;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Debug)]
pub struct ValueInner {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Value>,
    pub op: String,
    // pub backward: Rc<dyn Fn()>,
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            grad: 0f64,
            prev: Vec::with_capacity(8),
            op: String::with_capacity(8),
        }
    }
}

#[derive(Clone, Debug)]
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

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    pub fn backward(&mut self) {}
}

impl Add for Value {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let a = self.inner.borrow().data;
        let b = rhs.inner.borrow().data;
        let c = a + b;
        Value::new(c)
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.inner.borrow().data;
        let b = rhs.inner.borrow().data;
        let c = a * b;
        Value::new(c)
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
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::new(10.0);

        let d = a.clone() * b.clone();
        let e = d + c.clone();
        let f = Value::new(2.0);
        let mut l = e * f.clone(); // L = (a * b + c) * f = (2 * 3 + 10) * 2 = 32
        assert_eq!(l.data(), 32.0);

        l.backward();

        // dL/dl = 1
        // dL/df = e = 16
        // dL/de = f = 2
        // dL/dc = dL/de * 1 = 2
        // dL/dd = dL/de * 1 = 2
        // dL/da = dL/dd * b = 2 * 3 = 6
        // dL/db = dL/dd * a = 2 * 2 = 4

        assert_eq!(l.data(), 32.0);
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), 4.0);
        assert_eq!(c.grad(), 2.0);
        assert_eq!(f.grad(), 16.0);
    }
}
