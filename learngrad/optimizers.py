class SGD:
    def __init__(self, parameters, lr=0.01, lr_decay=0.0):
        self.lr_initial = lr
        self.lr = lr
        self.parameters = parameters
        self.lr_decay = lr_decay
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.lr = self.lr_initial / (1 + self.lr_decay * self.step_count)
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def __repr__(self):
        return f"SGD(lr={self.lr:.6f}, lr_initial={self.lr_initial}, lr_decay={self.lr_decay})"


class SGDMomentum:
    def __init__(self, parameters, lr=0.01, beta=0.9, lr_decay=0.0):
        self.lr_initial = lr
        self.lr = lr
        self.parameters = parameters
        self.beta = beta
        self.lr_decay = lr_decay
        self.v = [0] * len(parameters)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.lr = self.lr_initial / (1 + self.lr_decay * self.step_count)
        for i, p in enumerate(self.parameters):
            self.v[i] = self.v[i] * self.beta + p.grad
            p.data -= self.lr * self.v[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def __repr__(self):
        return f"SGDMomentum(lr={self.lr:.6f}, lr_initial={self.lr_initial}, beta={self.beta}, lr_decay={self.lr_decay})"


class RMSProp:
    def __init__(self, parameters, lr=0.01, beta=0.9, eps=1e-9, lr_decay=0.0) -> None:
        self.parameters = parameters
        self.lr_initial = lr
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.lr_decay = lr_decay
        self.s = [0] * len(parameters)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.lr = self.lr_initial / (1 + self.lr_decay * self.step_count)
        for i, p in enumerate(self.parameters):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * p.grad**2
            p.data -= self.lr * (p.grad / (self.s[i] ** 0.5 + self.eps))

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def __repr__(self):
        return f"RMSProp(lr={self.lr:.6f}, lr_initial={self.lr_initial}, beta={self.beta}, eps={self.eps}, lr_decay={self.lr_decay})"


class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.0):
        self.parameters = parameters
        self.lr_initial = lr
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_decay = lr_decay
        self.m = [0.0 for _ in parameters]
        self.v = [0.0 for _ in parameters]
        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def step(self):
        self.t += 1
        self.lr = self.lr_initial / (1 + self.lr_decay * self.t)
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

    def __repr__(self):
        return f"Adam(lr={self.lr:.6f}, lr_initial={self.lr_initial}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, lr_decay={self.lr_decay})"
