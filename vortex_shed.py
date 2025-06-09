import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import cv2
from utility.image_utility import stereo_transform

# Read and process velocity field data
x_vel, y_vel = [], []
x_pos, y_pos = [], []

with open('vortex/field_165.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        x_pos_t = float(row['Points:0']) 
        if 0.25 <= x_pos_t <= 0.75:
            x_vel.append(float(row['f_17-0:0']))
            y_vel.append(float(row['f_17-0:1']))
            x_pos.append(float(x_pos_t))
            y_pos.append(float(row['Points:1']))

x_pos, y_pos = np.array(x_pos), np.array(y_pos)
x_vel, y_vel = np.array(x_vel), np.array(y_vel)

x_vals = np.linspace(np.min(x_pos), np.max(x_pos), 256)
y_vals = np.linspace(np.min(y_pos), np.max(y_pos), 256)

grid_x, grid_y = np.meshgrid(x_vals, y_vals)
grid_x_vel = griddata((x_pos, y_pos), x_vel, (grid_x, grid_y), method='cubic')
grid_y_vel = griddata((x_pos, y_pos), y_vel, (grid_x, grid_y), method='cubic')
grid_x_vel = np.nan_to_num(grid_x_vel)
grid_y_vel = np.nan_to_num(grid_y_vel)

speed = np.sqrt(grid_x_vel**2 + grid_y_vel**2)

vorticity = np.zeros_like(grid_x_vel)
vorticity[1:-1, 1:-1] = (
    (grid_x_vel[:-2, 1:-1] - grid_x_vel[2:, 1:-1] + 
     grid_y_vel[1:-1, 2:] - grid_y_vel[1:-1, :-2]) / 2
)

# Create figure with 3 subplots
plt.figure(figsize=(10, 6))

# # X-velocity plot
# plt.subplot(1, 3, 1)
# plt.contourf(grid_x, grid_y, grid_x_vel, levels=20, cmap='coolwarm')
# plt.colorbar()
# plt.xlim(0.25, 0.75)
# plt.ylim(0, 0.41)
# plt.axis('off')

# Y-velocity plot
# plt.subplot(1, 3, 2)
# plt.contourf(grid_x, grid_y, grid_y_vel, levels=20, cmap='coolwarm')
# plt.colorbar()
# plt.xlim(0.25, 0.75)
# plt.ylim(0, 0.41)
# plt.axis('off')

# # Vorticity plot
# plt.subplot(1, 3, 3)
plt.contourf(grid_x, grid_y, vorticity, levels=20, cmap='coolwarm')
plt.colorbar()
plt.xlim(0.25, 0.75)
plt.ylim(0, 0.41)
plt.axis('off')

plt.tight_layout()
plt.savefig('velocity_w_fields.png', dpi=300)
# plt.show()

# plt.figure(figsize=(12, 8))

# plt.subplot(2, 2, 1)
# plt.scatter(x_pos, y_pos, c=np.sqrt(x_vel**2 + y_vel**2), s=5, cmap='rainbow')
# plt.colorbar(label='Speed (original points)')
# plt.title('Original Scattered Velocity Data')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.axis('equal')

# plt.subplot(2, 2, 2)
# plt.quiver(grid_x[::16,::16], grid_y[::16,::16], 
#            grid_x_vel[::16,::16], grid_y_vel[::16,::16],
#            speed[::16,::16], cmap='rainbow', scale=20)
# plt.colorbar(label='Speed (interpolated)')
# plt.title('Interpolated Velocity Field (Quiver)')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.axis('equal')

# plt.subplot(2, 2, 3)
# plt.contourf(grid_x, grid_y, speed, levels=20, cmap='rainbow')
# plt.colorbar(label='Speed Magnitude')
# plt.title('Speed Magnitude Contour')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.axis('equal')

# plt.subplot(2, 2, 4)
# try:
#     plt.streamplot(grid_x.T, grid_y.T, grid_x_vel.T, grid_y_vel.T, 
#                   color=speed.T, cmap='rainbow', density=1.5)
# except ValueError:
#     plt.streamplot(grid_x, grid_y, grid_x_vel, grid_y_vel, 
#                   color=speed, cmap='rainbow', density=1.5)
# plt.colorbar(label='Speed (streamlines)')
# plt.title('Velocity Streamlines')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.axis('equal')

# plt.savefig('vortex_field.png', dpi=300)
# plt.tight_layout()
# plt.show()

# src_img = "data/Synthetic_Data/Image/SNR_4/2/src.png"
# src_img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
# stereo_img = stereo_transform(src_img)

# height, width = src_img.shape[:2]

# deformation_scale = 2
# grid_x_vel *= deformation_scale
# grid_y_vel *= deformation_scale

# X, Y = np.meshgrid(np.arange(width), np.arange(height))
# map_x = (X - grid_x_vel).astype(np.float32)
# map_y = (Y - grid_y_vel).astype(np.float32)

# # Apply remapping
# deformed_img = cv2.remap(
#     src_img, map_x, map_y, 
#     interpolation=cv2.INTER_LINEAR, 
#     borderMode=cv2.BORDER_CONSTANT
#     )

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(src_img, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(deformed_img, cmap='gray')
# plt.title('Deformed Image (Remap)')
# plt.axis('off')

# plt.tight_layout()
# plt.show()