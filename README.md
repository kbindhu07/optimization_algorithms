# Optimization Algorithms from Scratch

## Technologies
- Python 3.x
- NumPy
- Matplotlib

## Objective
To implement and compare popular optimization algorithms from scratch and analyze
their convergence speed and stability on a convex optimization problem.

## Optimizers Implemented
- Stochastic Gradient Descent (SGD)
- Momentum
- Nesterov Accelerated Gradient
- AdaGrad
- RMSProp
- Adam

## Problem Statement
A 2D convex quadratic function is optimized:
f(x, y) = ax² + by²

The function has a global minimum at (0, 0), making it ideal for analyzing optimizer behavior.

## Experiments
- Optimized the function using each optimizer
- Recorded loss values for every iteration
- Plotted loss vs iteration curves
- Visualized optimization paths using contour plots

## Results
- SGD converges slowly and oscillates
- Momentum and Nesterov improve convergence speed
- AdaGrad converges quickly initially but slows down
- RMSProp provides stable convergence
- Adam achieves the fastest and most stable convergence

## Why This Project Matters
Understanding optimization algorithms is critical for training deep learning models
efficiently and avoiding slow or unstable convergence during training.

## How to Run
pip install -r requirements.txt  
python runner.py