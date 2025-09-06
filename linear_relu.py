import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Input space
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Linear combination
Z_linear = X + Y
# ReLU applied
Z_relu = np.maximum(0, Z_linear)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

contour = ax.contourf(X, Y, Z_linear, levels=20, cmap="coolwarm")
ax.set_title("Linear vs ReLU Transformation")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

def update(frame):
    ax.clear()
    if frame == 0:
        Z = Z_linear
        ax.set_title("Linear: z = x1 + x2")
    else:
        Z = Z_relu
        ax.set_title("ReLU: max(0, x1 + x2)")
    ax.contourf(X, Y, Z, levels=20, cmap="coolwarm")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

ani = FuncAnimation(fig, update, frames=2, interval=2000, repeat=True)
plt.show()
