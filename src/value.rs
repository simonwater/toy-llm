use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct Value {
    pub inner: Rc<RefCell<ValueInner>>,
}

pub struct ValueInner {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Value>,
    pub op: String,
    pub backward: Rc<dyn Fn()>,
}
