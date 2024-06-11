import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x,y):
    'Objective function'
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)


#Computing and plotting function in 3D within [0,10]x[0,10]
x , y = np.array(np.meshgrid(np.linspace(0,10,100) , np.linspace(0,10,100)))
z = f(x,y)

#golbal minima
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

#declaring Hyper parameters  
c1=0.1
c2=0.1
w=0.8
#now creating particles 
#seed value is an integer .. determine the subsequent random numbers.
n_particles = 30
np.random.seed(100)
X = np.random.rand(2 , n_particles)*10
# 2D array with dimensions (2, n_particles).
V = np.random.rand(2 , n_particles)*0.1

#initialize data
pbest = X
pbest_obj = f(X[0] ,X[1]) #more like f(x,y) which is random
gbest = pbest[: , pbest_obj.argmin()]
gbest_obj = pbest_obj.min()
#pbest_obj array that stores the objective function values (fitness values)

def update():
    "Function to do one iteration for PSO"
    global V,X,pbest_obj,pbest,gbest,gbest_obj #taking global values
    r1,r2 = np.random.rand(2)
    V = w*V + c1*r1*(pbest-X) + c2*r2*(gbest.reshape(-1,1)-X)
    X = X+V
    obj = f(X[0], X[1])  #fitness function
# Here, X[0]- x, and X[1] - y of all particles.    
    pbest[:, (pbest_obj >= obj)] = X[: ,(pbest_obj >= obj)] 
# This part creates a boolean mask using the condition pbest_obj >= obj.
#  The mask has the same dimensions as pbest (number of particles x 2 dimensions). 
    pbest_obj = np.array([pbest_obj,obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
#: This finds the index of the particle with the minimum objective function value
    gbest_obj = pbest_obj.min()
"""If a particle's current objective function value (obj) is lower than or equal to its personal best (pbest_obj), 
its personal best position in pbest gets updated with the particle's current position from X.
This ensures that each particle remembers its best position found so far during the optimization process."""

 # Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[0, 10, 0, 10], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='d', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([0,10])
ax.set_ylim([0,10])

def animate(i):
    "Steps of PSO: algorithm update and show in plot"
    title = 'Iteration {:02d}'.format(i)
    # Update params
    update()
    # Set picture
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])
    gbest_plot.set_offsets(gbest.reshape(1,-1))
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

anim = FuncAnimation(fig, animate, frames=list(range(1,50)), interval=500, blit=False, repeat=True)
anim.save("PSO.gif", dpi=120, writer="imagemagick")

print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
print("Global optimal at f({})={}".format([x_min,y_min], f(x_min,y_min)))  

    



