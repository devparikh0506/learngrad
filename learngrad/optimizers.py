class SGD:
    def __init__(self, parameters, lr=0.01):
        self.lr = lr
        self.parameters = parameters
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0


class SGDMomentum:
  def __init__(self, parameters, lr=0.01, beta=0.9):
        self.lr = lr
        self.parameters = parameters
        self.beta = beta
        self.v = [0] * len(parameters)
    
  def step(self):
      for i, p in enumerate(self.parameters):
          self.v[i] = self.v[i] * self.beta + p.grad
          p.data -= self.lr * self.v[i]
  
  def zero_grad(self):
      for p in self.parameters:
          p.grad = 0

class RMSProp:
  def __init__(self, parameters, lr=0.01, beta=0.9, eps = 1e-9) -> None:
     self.parameters = parameters
     self.lr = lr
     self.beta = beta
     self.eps = eps
     self.s = [0] * len(parameters)

  def step(self):
    for i, p in enumerate(self.parameters):
          self.s[i] = self.beta * self.s[i] + (1 - self.beta) * p.grad**2
          p.data -= self.lr * (p.grad / (self.s[i] ** 0.5 + self.eps))
  def zero_grad(self):
      for p in self.parameters:
          p.grad = 0


class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0 for _ in parameters]
        self.v = [0.0 for _ in parameters]
        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)