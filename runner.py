import numpy as np
import matplotlib.pyplot as plt
from optimizers import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam

a, b = 1.0, 10.0

def loss(params):
    x, y = params
    return a * x**2 + b * y**2

def gradient(params):
    x, y = params
    return np.array([2 * a * x, 2 * b * y])

optimizers = {
    "SGD": SGD(lr=0.05),
    "Momentum": Momentum(lr=0.05),
    "Nesterov": Nesterov(lr=0.05),
    "AdaGrad": AdaGrad(lr=0.5),
    "RMSProp": RMSProp(lr=0.05),
    "Adam": Adam(lr=0.05)
}

iterations = 100
start_point = np.array([5.0, 5.0])

history = {}
paths = {}

for name, optimizer in optimizers.items():
    params = start_point.copy()
    losses = []
    path_x, path_y = [], []

    for _ in range(iterations):
        path_x.append(params[0])
        path_y.append(params[1])

        grads = gradient(params)
        params = optimizer.step(params, grads)
        losses.append(loss(params))

    history[name] = losses
    paths[name] = (path_x, path_y)

    print(f"{name} final params: {params}")

plt.figure(figsize=(10, 6))
for name, losses in history.items():
    plt.plot(losses, label=name)

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Convergence Comparison of Optimization Algorithms")
plt.legend()
plt.grid(True)

x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
Z = a * X**2 + b * Y**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30)

for name, (px, py) in paths.items():
    plt.plot(px, py, marker='o', markersize=2, label=name)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Optimization Paths on Loss Contour")
plt.legend()
plt.grid(True)

plt.show()