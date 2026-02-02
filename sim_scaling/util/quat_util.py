import numpy as np
import quaternion

def make_quat(deg, axis):
    rad = np.deg2rad(deg)
    axis = axis / np.linalg.norm(axis)
    return quaternion.from_rotation_vector(rad * axis)