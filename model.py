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

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    def forward(x):
      y = np.maximum(x,0.0)
      cache = x
      return y, cache 
      
    def backward(dout, cache):
      dx = dout * (cache>0)
      return dx, {}

    return {
      'params': {},
      'forward': forward,
      'backward': backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    s = np.sqrt(2/in_dim)
    W = np.random.normal(0.0, s,(in_dim, out_dim))
    b = np.zeros((out_dim,))
    return W,b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
  """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

  Inputs to loss_fn:
    logits: (batch, C) float array of raw class scores
    labels: (batch,) int array of class indices in [0, C)
  Outputs:
    loss: Python float, mean scalar loss over the batch (finite)
    d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
  Must pass gradient_check, be minimized by confident correct predictions,
  and stay finite under saturated logits.
  """
  def loss_fn(logits, labels):
      B = logits.shape[0]
      shifted = logits - logits.max(axis=1, keepdims=True)        # stability
      log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
      loss = -log_probs[np.arange(B), labels].mean()

      probs = np.exp(log_probs)
      d_logits = probs.copy()
      d_logits[np.arange(B), labels] -= 1.0
      d_logits /= B

      return float(loss), d_logits

  return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    def forward(x):
      caches = []
      for layer in layers:
        x, cache = layer['forward'](x)
        caches.append(cache)
      return x, caches  

    def backward(dout, caches):
        grads_list = []
        for layer, cache in zip(reversed(layers), reversed(caches)):
            dout, grads = layer['backward'](dout, cache)
            grads_list.append(grads)
        grads_list.reverse()          # back to layer order
        return dout, grads_list

    params = []
    for layer in layers:
      params.append(layer['params'])
      
    return {
      'forward': forward,
      'backward': backward,
      'params': params
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    logits, caches = model['forward'](x)
    loss, d_logits = loss_fn(logits, y)
    dx, param_grads = model['backward'](d_logits, caches)
    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    def _update(p, g):
        if isinstance(p, dict):
            for k in p:
                _update(p[k], g[k])
        elif isinstance(p, (list, tuple)):
            for pi, gi in zip(p, g):
                _update(pi, gi)
        else:
            p -= lr * g

    def step(grads):
        _update(params, grads)

    return {'step': step}

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

