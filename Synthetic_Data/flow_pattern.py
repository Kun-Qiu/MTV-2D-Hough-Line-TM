import numpy as np
import matplotlib.pyplot as plt
import cv2

def poiseuille_flow(U, shape):
    """
    Generate and visualize the displacement field for Poiseuille flow.

    :param U        : Maximum velocity (displacement intensity)
    :param image    : Input image array
    :return         : Displacement field (dx, dy)
    """

    h, w = shape
    h_half = h // 2

    # Generate displacement field using Poiseuille flow profile
    y = np.linspace(-h_half, h_half, h)
    u_x = U * (1 - (y**2 / h_half**2)) 

    """
    Displacement in x and no displacemnt in y
    """
    displacement_x = np.tile(u_x[:, np.newaxis], (1, w)).astype(np.float32)
    displacement_y = np.zeros_like(displacement_x, dtype=np.float32)  

    """
    Visualization - Uncomment
    """
    
    # X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # # Subsample for clarity
    # sampling = 20  # Increase sampling interval to improve visualization
    # X_sub, Y_sub = X[::sampling, ::sampling], Y[::sampling, ::sampling]
    # dx_sub, dy_sub = displacement_x[::sampling, ::sampling], displacement_y[::sampling, ::sampling]

    # plt.figure(figsize=(8, 6))
    # plt.quiver(X_sub, Y_sub, dx_sub, dy_sub, 
    #            scale=1, color='r', angles='xy', scale_units='xy')
    # plt.title("Poiseuille Flow Displacement Field")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    # plt.grid(False)
    # plt.show()

    return np.stack((displacement_x, displacement_y), axis=-1)  


def lamb_oseen_vortex(gamma=0.05, sqrt_4_nu_t = 1 / 6, scale=250, shape=(256, 256)):
    """
    Generate and visualize the velocity field of the Lamb-Oseen vortex.

    :param gamma    :   Circulation coefficient [m^2/s]
    :param shape    :   Tuple specifying image dimensions (height, width)
    :return         :   Array where each element is [u_r, u_theta]
    """
    # Given parameters
    nu_t = (sqrt_4_nu_t ** 2) / 4  # Derived value 4νt = (1/6)^2

    # Grid setup (centered at origin)
    h, w = shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Avoid division by zero by replacing r=0 with a small value
    r[r == 0] = 1e-8

    # Calculate u_theta (tangential velocity)
    u_theta = (gamma / (2 * np.pi * r)) * (1 - np.exp(-r**2 / (4 * nu_t)))
    u_x = scale * -u_theta * np.sin(theta)
    u_y = scale * u_theta * np.cos(theta)

    # Visualization of the velocity field using colormap
    # plt.figure(figsize=(10, 5))
    
    # # Magnitude of velocity for colormap
    # velocity_magnitude = np.sqrt(u_x**2 + u_y**2)
    # plt.imshow(velocity_magnitude, cmap='jet', extent=[-1, 1, -1, 1])
    # plt.colorbar(label='Velocity Magnitude')
    # plt.title("Lamb-Oseen Vortex Velocity Field")
    # plt.xlabel("X-axis (m)")
    # plt.ylabel("Y-axis (m)")
    
    # # Overlay vector field using quiver
    # sampling = 10  # Reduce the number of arrows for better visualization
    # plt.quiver(X[::sampling, ::sampling], Y[::sampling, ::sampling], 
    #            u_x[::sampling, ::sampling], u_y[::sampling, ::sampling], 
    #            scale=25, color='w')
    
    # plt.grid(False)
    # plt.show()

    return np.stack((u_x, u_y), axis=-1)


def uniform_flow(x_translate, y_translate, shape=(256, 256)):
    """
    Determine the x and y translation displacement field

    :param shape        :   The input image of which the transformation is performed
    :param x_translate  :   The number of pixel to translate in the x direction
    :param y_translate  :   The number of pixel to translate in the y direction
    :return:                The displacement field
    """

    h, w = shape

    # Displacement Field
    displacement_field = np.zeros((h, w, 2))
    displacement_field[..., 0] = x_translate
    displacement_field[..., 1] = y_translate

    # X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    u = displacement_field[..., 0]  # x displacement
    v = displacement_field[..., 1]  # y displacement

    # plt.figure(figsize=(8, 6))
    # plt.quiver(X, Y, u, v, angles='xy', scale_units='xy', scale=1, color='blue')
    # plt.title("Uniform Flow")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.xlim(-1, shape[1])
    # plt.ylim(-1, shape[0])
    # plt.gca().invert_yaxis()  # To match image coordinate system
    # plt.show()
    
    return displacement_field


if __name__ == "__main__":
    # poiseuille_dit = poiseuille_flow(U=10, shape=(256,256))
    vortex_dist = lamb_oseen_vortex()
    # uniform_dist = uniform_flow(2, 1)