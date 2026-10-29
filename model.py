"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps 
        f_minus = f(x)
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x[idx] = orig 

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic_grad = np.asarray(analytic_grad)
    numeric_grad = np.asarray(numeric_grad)
    abs_diff = np.abs(analytic_grad - numeric_grad)
    elementwise_max = np.maximum(np.abs(analytic_grad), np.abs(numeric_grad))
    elementwise_max = np.maximum(elementwise_max, tol)
    return float(np.max(abs_diff / elementwise_max))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W,b = weight_init_fn(in_dim, out_dim)
    params = {'W':W, 'b':b}

    def forward(x):
      y = x @ params['W'] + params['b']
      cache = x
      return y, cache 

    def backward(dout, cache):
      x = cache 
      dW = x.T @ dout 
      db = dout.sum(axis=0)
      dx = dout @ W.T
      return dx, {'W': dW, 'b': db}

    return {'params': params, 'forward': forward, 'backward': backward}

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

