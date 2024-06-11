from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    """Objective function"""
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)


# Computing and plotting function in 3D within [0, 10] x [0, 10]
x, y = np.array(np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100)))
z = f(x, y)

# Global minima
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

# Hyper parameters
c1 = 0.1
c2 = 0.1
w = 0.8

# Particles
n_particles = 30
np.random.seed(100)
X = np.random.rand(2, n_particles) * 10
V = np.random.rand(2, n_particles) * 0.1

# Personal bests
pbest = X.copy()
pbest_obj = f(X[0], X[1])

# Global best
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()


def update():
    """Function to do one iteration for PSO"""
    global V, X, pbest_obj, pbest, gbest, gbest_obj
    r1, r2 = np.random.rand(2)
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
    X = X + V
    obj = f(X[0], X[1])

    # Update personal bests
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)

    # Update global best
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()


def generate_frame(i):
    """Generates a frame for the animation"""
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # Plot contour map
    img = ax.imshow(
        z, extent=[0, 10, 0, 10], origin="lower", cmap="viridis", alpha=0.5
    )
    fig.colorbar(img, ax=ax)

    # Plot minimum
    ax.plot([x_min], [y_min], marker="d", markersize=5, color="white")

    # Plot contours
    contours = ax.contour(x, y, z, 10, colors="black", alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    # Plot particles
    ax.scatter(pbest[0], pbest[1], marker="o", color="black", alpha=0.5, label="Personal Best")
    p_plot = ax.scatter(X[0], X[1], marker="o", color="blue", alpha=0.5, label="Particles")

    # Plot velocity arrows
    p_arrow = ax.quiver(
        X[0], X[1], V[0], V[1], color="blue", width=0.005, angles="xy", scale_units="xy", scale=1
    )

    # Plot global best
    gbest_plot = plt.scatter(
        [gbest[0]], [gbest[1]], marker="*", s=100, color="black", alpha=0.4, label="Global Best"
    )

    ax.set_title(f"Iteration {i:02d}")
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.legend()

    # Convert plot to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer
