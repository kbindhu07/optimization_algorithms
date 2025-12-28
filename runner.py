import numpy as np
import matplotlib.pyplot as plt
from optimizers import sgd, momentum, nesterov, adagrad, rmsprop, adam

# -----------------------------
# Run optimizers
# -----------------------------
results = {
    "SGD": sgd(),
    "Momentum": momentum(),
    "Nesterov": nesterov(),
    "AdaGrad": adagrad(),
    "RMSProp": rmsprop(),
    "Adam": adam()
}

# -----------------------------
# 1️⃣ Convergence Plot
# -----------------------------
plt.figure(figsize=(8, 6))

for name, (losses, _) in results.items():
    plt.plot(losses, label=name)

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Convergence Plot of Optimizers")
plt.legend()
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 2️⃣ Contour Plot
# -----------------------------
x = np.linspace(-11, 11, 400)
y = np.linspace(-11, 11, 400)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30)
plt.title("Contour Plot with Optimizer Paths")

for name, (_, path) in results.items():
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], marker='o', label=name)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.savefig("contour_plot.png", dpi=300, bbox_inches="tight")
plt.show()