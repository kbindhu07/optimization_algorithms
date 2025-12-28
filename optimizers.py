import numpy as np

def loss_fn(x, y):
    return x**2 + y**2

def grad_fn(x, y):
    return 2*x, 2*y


def sgd(lr=0.1, steps=50):
    x, y = 10.0, 10.0
    losses, path = [], []

    for _ in range(steps):
        gx, gy = grad_fn(x, y)
        x -= lr * gx
        y -= lr * gy
        losses.append(loss_fn(x, y))
        path.append((x, y))

    return losses, path


def momentum(lr=0.1, beta=0.9, steps=50):
    x, y = 10.0, 10.0
    vx, vy = 0.0, 0.0
    losses, path = [], []

    for _ in range(steps):
        gx, gy = grad_fn(x, y)
        vx = beta * vx + lr * gx
        vy = beta * vy + lr * gy
        x -= vx
        y -= vy
        losses.append(loss_fn(x, y))
        path.append((x, y))

    return losses, path


def nesterov(lr=0.1, beta=0.9, steps=50):
    x, y = 10.0, 10.0
    vx, vy = 0.0, 0.0
    losses, path = [], []

    for _ in range(steps):
        gx, gy = grad_fn(x - beta*vx, y - beta*vy)
        vx = beta * vx + lr * gx
        vy = beta * vy + lr * gy
        x -= vx
        y -= vy
        losses.append(loss_fn(x, y))
        path.append((x, y))

    return losses, path


def adagrad(lr=1.0, eps=1e-8, steps=50):
    x, y = 10.0, 10.0
    hx, hy = 0.0, 0.0
    losses, path = [], []

    for _ in range(steps):
        gx, gy = grad_fn(x, y)
        hx += gx**2
        hy += gy**2
        x -= lr * gx / (np.sqrt(hx) + eps)
        y -= lr * gy / (np.sqrt(hy) + eps)
        losses.append(loss_fn(x, y))
        path.append((x, y))

    return losses, path


def rmsprop(lr=0.1, beta=0.9, eps=1e-8, steps=50):
    x, y = 10.0, 10.0
    hx, hy = 0.0, 0.0
    losses, path = [], []

    for _ in range(steps):
        gx, gy = grad_fn(x, y)
        hx = beta * hx + (1-beta) * gx**2
        hy = beta * hy + (1-beta) * gy**2
        x -= lr * gx / (np.sqrt(hx) + eps)
        y -= lr * gy / (np.sqrt(hy) + eps)
        losses.append(loss_fn(x, y))
        path.append((x, y))

    return losses, path


def adam(lr=0.1, b1=0.9, b2=0.999, eps=1e-8, steps=50):
    x, y = 10.0, 10.0
    mx = my = vx = vy = 0.0
    losses, path = [], []

    for t in range(1, steps+1):
        gx, gy = grad_fn(x, y)

        mx = b1*mx + (1-b1)*gx
        my = b1*my + (1-b1)*gy
        vx = b2*vx + (1-b2)*gx**2
        vy = b2*vy + (1-b2)*gy**2

        mx_hat = mx / (1-b1**t)
        my_hat = my / (1-b1**t)
        vx_hat = vx / (1-b2**t)
        vy_hat = vy / (1-b2**t)

        x -= lr * mx_hat / (np.sqrt(vx_hat) + eps)
        y -= lr * my_hat / (np.sqrt(vy_hat) + eps)

        losses.append(loss_fn(x, y))
        path.append((x, y))

    return losses, path