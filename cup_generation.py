import numpy as np
from scipy.spatial.transform import Rotation as R

radius = 0.05
height = 0.061
# bottom_height = 0.026
num_faces = 20

unit_vec = np.array([radius, 0, 0])
unit_rad = 2 * np.pi / num_faces

for i in range(num_faces):
    rad = i * unit_rad
    rot = R.from_euler('xyz', [0, 0, rad])
    vec = rot.apply(unit_vec)
    # print(vec, rad)
    print(f'<geom class="cup" pos="{vec[0]:.5f} {vec[1]:.5f} {height / 2:.5f}" euler="0 0 {rad:.5f}"/>')

print(f'<geom class="cup" type="cylinder" size="{radius:.3f} 0.0026" pos="0 0 0.0026"/>')

face_len = radius * np.sin(unit_rad / 2)
print(f"{face_len:.6f} {height / 2:.5f}")