# Made in Collaboration by Pranav P and Jack G, Imperial College London Mechanical Engineering

"""
TASK 0: Import necessary libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

"""
TASK 1: Choose a physical problem
"""

# Object being dropped into a pool

"""
TASK 2: Set up an apropriate PDE
"""

# Equation is u_tt = c^2 * (u_xx + u_yy)

"""
TASK 3: Set up boundary and initial conditions
"""

def initial_condition(x, y, x0, y0, a):
    # Gaussian initial condition centered at (x0, y0) with spread controlled by a
    return 0.05*np.exp(-a * ((x - x0) ** 2 + (y - y0) ** 2))

def boundary_condition(u):
    # Neumann boundary conditions with zero gradient at the edges
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    return u

"""
TASK 4: Choose a method to numerically solve the PDE
"""

# Finite Difference Method (FDM)

"""
TASK 5: Discretise the PDE
"""

# We applied finite differences to discretise the PDE

"""
TASK 6: Solve the PDE
"""

def solve_wave_equation(Nx, Ny, Nt, c, x0, y0, a, dx, dy, dt):
    """
    Solves the wave equation using a finite difference method.

    Parameters:
    - Nx, Ny: Number of grid points in the x and y directions.
    - Nt: Number of time steps.
    - c: Speed of wave propagation.
    - x0, y0: Coordinates of the initial disturbance.
    - a: Parameter controlling the spread of the initial Gaussian.
    - dx, dy: Spatial step sizes.
    - dt: Time step size.
    """
    # Initialize solution grid
    u = np.zeros((Nt, Nx, Ny), dtype='float' )
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    # Apply initial condition
    u[0, :, :] = initial_condition(X, Y, x0, y0, a)

    # Initial velocity = 0 implies u[1,:,:] = u[0,:,:]
    u[1, :, :] = u[0, :, :]

    # Finite difference coefficients
    coeff = (c * dt / dx) ** 2

    # Time evolution
    for n in range(1, Nt - 1):
        u[n + 1, 1:-1, 1:-1] = (2 * u[n, 1:-1, 1:-1] - u[n - 1, 1:-1, 1:-1] +
                                coeff * (u[n, 2:, 1:-1] - 2 * u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1] +
                                         u[n, 1:-1, 2:] - 2 * u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2]))
        u[n + 1] = boundary_condition(u[n + 1])

    return u

"""
TASK 7: Create a discretised grid and apply the PDE solver
"""

# Create the discretised grid
Nx, Ny, Nt = 200, 200, 2000  # Grid size and number of time steps
c = 1  # Wave speed
x0, y0 = 0.5, 0.5  # Initial disturbance center
a = 1000  # Spread of the initial Gaussian
dx, dy = (1 / (Nx - 1)), ( 1 / (Ny - 1) ) # Spatial step sizes
dt = 0.001  # Time step size
print(c*dt/dx)
# Solve the wave equation
u = solve_wave_equation(Nx, Ny, Nt, c, x0, y0, a, dx, dy, dt)


"""
TASK 8: Plot the results of the solving
"""

#plotting the initial conditions
x = np.linspace(-x0, x0, Nx)
y = np.linspace(-y0, y0, Ny)
x, y = np.meshgrid(x, y)
z = u[0, :, :]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(x, y, z, cmap='viridis')

# Add a color bar which maps values to colors
fig.colorbar(surf)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Displacement')

# Show plot
plt.show()




# Assuming 'u' is your 3D array containing the wave displacements over time
# And assuming 'u' has dimensions [time, x, y], and you've already solved the wave equation to populate 'u'

# Initialize the figure and axes for the animation, ensuring axes range from 0 to 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((-x0, x0))
ax.set_ylim((-y0, y0))
ax.set_zlim((-0.05, 0.05))

x = np.linspace(-x0, x0, Nx)
y = np.linspace(-y0, y0, Ny)
x, y = np.meshgrid(x, y)

def update(frame_number, u, plot):
   plot[0].remove()
   plot[0] = ax.plot_surface(x, y, u[frame_number, :, :], cmap="viridis")


plot = [ax.plot_surface(x, y, u[0, :, :], cmap="viridis")]


ani = FuncAnimation(fig, update, frames=range(1,u.shape[0],10),fargs=(u, plot), blit=False, interval = 1)




f = r"c://Users/Jack//OneDrive/Desktop/3D_wave_animation.gif" 
writergif = animation.PillowWriter(fps=10) 
ani.save(f, writer=writergif)
# Display the animation
#plt.show(ani)


# Plotting the heat maps 

# Initialize the figure and axes for the animation, ensuring axes range from 0 to 1
fig, ax = plt.subplots()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))

# Initial plot is a heatmap of the first time step
heatmap = ax.imshow(u[0, :, :], extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')

# Function to update the heatmap for each frame
def update(frame):
    heatmap.set_data(u[frame, :, :])
    return [heatmap]

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0,u.shape[0],10), blit=False, interval = 10)


f = r"c://Users/Jack//OneDrive/Desktop/Heatmap_200x200_animation.gif" 
writergif = animation.PillowWriter(fps=10) 
ani.save(f, writer=writergif)
# Display the animation
#plt.show(ani)





"""
TASK 9: Repeat the process using a 20x20 grid
"""

# Create the discretised grid
Nx, Ny, Nt = 20, 20, 2000  # Grid size and number of time steps
c = 1  # Wave speed
x0, y0 = 0.5, 0.5  # Initial disturbance center
a = 1000  # Spread of the initial Gaussian
dx, dy = (1 / (Nx - 1)), ( 1 / (Ny - 1) ) # Spatial step sizes
dt = 0.001  # Time step size
print(c*dt/dx)
# Solve the wave equation
u = solve_wave_equation(Nx, Ny, Nt, c, x0, y0, a, dx, dy, dt)


# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(x, y, z, cmap='viridis')

# Add a color bar which maps values to colors
fig.colorbar(surf)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Displacement')

# Show plot
#plt.show()




# Assuming 'u' is your 3D array containing the wave displacements over time
# And assuming 'u' has dimensions [time, x, y], and you've already solved the wave equation to populate 'u'

# Plotting the heat maps 

# Initialize the figure and axes for the animation, ensuring axes range from 0 to 1
fig, ax = plt.subplots()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))

# Initial plot is a heatmap of the first time step
heatmap = ax.imshow(u[0, :, :], extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')

# Function to update the heatmap for each frame
def update(frame):
    heatmap.set_data(u[frame, :, :])
    return [heatmap]

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0,u.shape[0],10), blit=False, interval = 10)

f = r"c://Users/Jack//OneDrive/Desktop/Heatmap_20x20_animation.gif" 
writergif = animation.PillowWriter(fps=10) 
ani.save(f, writer=writergif)
# Display the animation
#plt.show(ani)