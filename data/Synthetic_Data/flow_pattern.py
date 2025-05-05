import numpy as np
import matplotlib.pyplot as plt

def poiseuille_flow(U, shape=(256, 256), filename=None, show=False):
    """
    Generate and visualize the displacement field for Poiseuille flow.

    :param U        :   Maximum velocity
    :param image    :   Input image array (default = 256 x 256)
    :filename       :   If filename is not none, save the displacement to local directory
    :show           :   Boolean to show the displacement field using matplotlib
    :return         :   Displacement field (dx, dy)
    """

    h, w = shape
    h_half = h // 2

    # Generate displacement field using Poiseuille flow profile
    y = np.linspace(-h_half, h_half, h)
    u_x = U * (1 - (y**2 / h_half**2)) 

    displacement_x = np.tile(u_x[:, np.newaxis], (1, w)).astype(np.float32)
    displacement_y = np.zeros_like(displacement_x, dtype=np.float32)
    field = np.stack((displacement_x, displacement_y), axis=-1) 

    if show:
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # Subsample for clarity
        sampling = 20  # Increase sampling interval to improve visualization
        X_sub, Y_sub = X[::sampling, ::sampling], Y[::sampling, ::sampling]
        dx_sub, dy_sub = displacement_x[::sampling, ::sampling], displacement_y[::sampling, ::sampling]

        plt.figure(figsize=(8, 6))
        plt.quiver(X_sub, Y_sub, dx_sub, dy_sub, 
                scale=1, color='r', angles='xy', scale_units='xy')
        plt.title("Poiseuille Flow Displacement Field")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.grid(False)
        plt.show()

    if filename != None:
        np.save(filename, field)
        print(f"Velocity field saved to {filename}")

    return field


def lamb_oseen_vortex(shape=(256, 256), gamma=0.05, sqrt_4_nu_t = 1 / 6, scale=250, 
                      filename=None, show=False):
    """
    Generate and visualize the velocity field of the Lamb-Oseen vortex.

    :param shape    :   Tuple specifying image dimensions (height, width)
    :param gamma    :   Circulation coefficient [m^2/s]
    :scale          :   Scaling factor of rotation
    :filename       :   If filename is not none, save the displacement to local directory
    :show           :   Boolean to show the displacement field using matplotlib
    :return         :   Array where each element is [u_r, u_theta]
    """
    nu_t = (sqrt_4_nu_t ** 2) / 4  # Derived value 4νt = (1/6)^2

    h, w = shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    r[r == 0] = 1e-8

    # Calculate u_theta (tangential velocity)
    u_theta = (gamma / (2 * np.pi * r)) * (1 - np.exp(-r**2 / (4 * nu_t)))
    u_x = scale * -u_theta * np.sin(theta)
    u_y = scale * u_theta * np.cos(theta)

    # Visualization of the velocity field using colormap
    if show:
        plt.figure(figsize=(10, 5))
        
        velocity_magnitude = np.sqrt(u_x**2 + u_y**2)
        plt.imshow(velocity_magnitude, cmap='jet', extent=[-1, 1, -1, 1])
        plt.colorbar(label='Velocity Magnitude')
        plt.title("Lamb-Oseen Vortex Velocity Field")
        plt.xlabel("X-axis (m)")
        plt.ylabel("Y-axis (m)")
        
        sampling = 10 
        plt.quiver(X[::sampling, ::sampling], Y[::sampling, ::sampling], 
                u_x[::sampling, ::sampling], u_y[::sampling, ::sampling], 
                scale=25, color='w')
        
        plt.grid(False)
        plt.show()

    field = np.stack((u_x, u_y), axis=-1)
    if filename != None:
        np.save(filename, field)
        print(f"Velocity field saved to {filename}")

    return field


def uniform_flow(x_translate, y_translate, shape=(256, 256), filename=None, show=False):
    """
    Determine the x and y translation displacement field

    :param shape        :   The input image of which the transformation is performed
    :param x_translate  :   The number of pixel to translate in the x direction
    :param y_translate  :   The number of pixel to translate in the y direction
    :filename           :   If filename is not none, save the displacement to local directory
    :show               :   Boolean to show the displacement field using matplotlib
    :return:            :   The displacement field
    """

    h, w = shape

    # Displacement Field
    field = np.zeros((h, w, 2))
    field[..., 0] = x_translate
    field[..., 1] = y_translate

    if show:
        X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        u = field[..., 0]  
        v = field[..., 1]  

        plt.figure(figsize=(8, 6))
        
        sampling = 20  # Increase sampling interval to improve visualization
        X_sub, Y_sub = X[::sampling, ::sampling], Y[::sampling, ::sampling]
        dx_sub, dy_sub = u[::sampling, ::sampling], v[::sampling, ::sampling]

        plt.quiver(X_sub, Y_sub, dx_sub, dy_sub, angles='xy', scale_units='xy', scale=1, color='blue')
        plt.title("Uniform Flow")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.xlim(-1, shape[1])
        plt.ylim(-1, shape[0])
        plt.gca().invert_yaxis()
        plt.show()
    
    if filename != None:
        np.save(filename, field)
        print(f"Velocity field saved to {filename}")
    
    return field


if __name__ == "__main__":
    # poiseuille_dit = poiseuille_flow(U=10, shape=(256,256))
    vortex_dist = lamb_oseen_vortex()
    # uniform_dist = uniform_flow(2, 1)