import sim_scaling.policy.base_policy
import torch
import numpy as np
from isaaclab.utils import math as tensor_math

def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert batched quaternions to rotation matrices.
    Args:
        q: (..., 4) tensor of quaternions in (w, x, y, z) order.
    Returns:
        R: (..., 3, 3) rotation matrices.
    """
    # Normalize to handle non-unit inputs
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    # Rotation matrix entries (right-handed, active, column-major equivalent)
    # R = [[1-2(yy+zz), 2(xy-wz),   2(xz+wy)],
    #      [2(xy+wz),   1-2(xx+zz), 2(yz-wx)],
    #      [2(xz-wy),   2(yz+wx),   1-2(xx+yy)]]
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R
    
class ControlPolicyGPU:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.key_point_poses = [
            np.array([0.000, 0.055, 0.000]), # 0
            np.array([0.025, 0.055, 0.000]), # 1
            np.array([0.050, 0.055, 0.000]), # 2
            np.array([0.075, 0.055, 0.000]), # 3
            np.array([0.075, 0.055, 0.000]), # 4
            np.array([0.075, 0.055, 0.000]), # 5
            np.array([0.075, 0.030, 0.000]), # 6
            np.array([0.075, 0.005, 0.000]), # 7
            np.array([0.075, 0.005, 0.000]), # 8
            np.array([0.075, 0.005, 0.000]), # 9
            np.array([0.050, 0.005, 0.000]), # 10
            np.array([0.025, -0.020, 0.000]), # 11
            np.array([0.025, -0.045, 0.000]), # 12
            np.array([0.025, -0.070, 0.000]), # 13
            np.array([0.025, -0.095, 0.000]), # 14
            np.array([0.025, -0.095, 0.000]), # 15
            np.array([0.025, -0.095, 0.000]), # 16
            np.array([0.000, -0.095, 0.000]), # 17
            np.array([-0.025, -0.095, 0.000]), # 18
            np.array([-0.025, -0.095, 0.000]), # 19
            np.array([-0.025, -0.095, 0.000]), # 20
            np.array([-0.025, -0.070, 0.000]), # 21
            np.array([-0.025, -0.045, 0.000]), # 22
            np.array([-0.025, -0.020, 0.000]), # 23
            np.array([-0.050, 0.005, 0.000]), # 24
            np.array([-0.075, 0.005, 0.000]), # 25
            np.array([-0.075, 0.005, 0.000]), # 26
            np.array([-0.075, 0.005, 0.000]), # 27
            np.array([-0.075, 0.030, 0.000]), # 28
            np.array([-0.075, 0.055, 0.000]), # 29
            np.array([-0.075, 0.055, 0.000]), # 30
            np.array([-0.075, 0.055, 0.000]), # 31
            np.array([-0.050, 0.055, 0.000]), # 32
            np.array([-0.025, 0.055, 0.000]), # 33
        ]

        self.key_point_normals = [
            np.array([0.0, 1.0, 0.0]), # 0
            np.array([0.0, 1.0, 0.0]), # 1
            np.array([0.0, 1.0, 0.0]), # 2
            np.array([0.0, 1.0, 0.0]), # 3
            np.array([1.0, 1.0, 0.0]), # 4
            np.array([1.0, 0.0, 0.0]), # 5
            np.array([1.0, 0.0, 0.0]), # 6
            np.array([1.0, 0.0, 0.0]), # 7
            np.array([1.0, -1.0, 0.0]), # 8
            np.array([0.0, -1.0, 0.0]), # 9
            np.array([0.0, -1.0, 0.0]), # 10
            np.array([1.0, 0.0, 0.0]), # 11
            np.array([1.0, 0.0, 0.0]), # 12
            np.array([1.0, 0.0, 0.0]), # 13
            np.array([1.0, 0.0, 0.0]), # 14
            np.array([1.0, -1.0, 0.0]), # 15
            np.array([0.0, -1.0, 0.0]), # 16
            np.array([0.0, -1.0, 0.0]), # 17
            np.array([0.0, -1.0, 0.0]), # 18
            np.array([-1.0, -1.0, 0.0]), # 19
            np.array([-1.0, 0.0, 0.0]), # 20
            np.array([-1.0, 0.0, 0.0]), # 21
            np.array([-1.0, 0.0, 0.0]), # 22
            np.array([-1.0, 0.0, 0.0]), # 23
            np.array([0.0, -1.0, 0.0]), # 24
            np.array([0.0, -1.0, 0.0]), # 25
            np.array([-1.0, -1.0, 0.0]), # 26
            np.array([-1.0, 0.0, 0.0]), # 27
            np.array([-1.0, 0.0, 0.0]), # 28
            np.array([-1.0, 0.0, 0.0]), # 29
            np.array([-1.0, 1.0, 0.0]), # 30
            np.array([0.0, 1.0, 0.0]), # 31
            np.array([0.0, 1.0, 0.0]), # 32
            np.array([0.0, 1.0, 0.0]), # 33
        ]
        
        self.key_point_poses = torch.tensor(self.key_point_poses, dtype=torch.float32, device=device)
        self.key_point_normals = torch.tensor(self.key_point_normals, dtype=torch.float32, device=device)
        self.key_point_normals = self.key_point_normals / torch.norm(self.key_point_normals, dim=1, keepdim=True)

        # store last chosen keypoint per env
        self.last_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def control(self, head_pose, pose, targ_pose):
        pos = pose[:, :3]
        quat = pose[:, 3:7]
        R = quat_to_mat(quat)
        head_pos = head_pose[:, :3]
        targ_pos = targ_pose[:3]
        targ_quat = targ_pose[3:7]
        # make targ_pose and targ_quat batched
        targ_pos = targ_pos[None, :].repeat(pos.shape[0], 1)
        targ_quat = targ_quat[None, :].repeat(pos.shape[0], 1)

        rel_pos = head_pos - pos

        rel_pos_in_obj = R.transpose(-1, -2) @ rel_pos.unsqueeze(-1)
        rel_pos_in_obj = rel_pos_in_obj.squeeze(-1)
        move_direction = -rel_pos_in_obj.clone()

        tpos_diff = (targ_pos - pos) / 0.1
        tpos_diff_in_obj = R.transpose(-1, -2) @ tpos_diff.unsqueeze(-1)
        tpos_diff_in_obj = tpos_diff_in_obj.squeeze(-1)
        tquat_diff = tensor_math.quat_mul(tensor_math.quat_conjugate(targ_quat), quat)
        tangle_diff = tensor_math.axis_angle_from_quat(tquat_diff)[:, 2]

        # Precompute keypoint-dependent constants (K,3)
        # f = -normals
        f = -self.key_point_normals  # (K,3)
        r = self.key_point_poses  # (K,3)
        tau = torch.cross(r / 0.1, f, dim=1)  # (K,3)
        omega = tau / 0.3  # (K,3)
        tau_z = tau[:, 2]  # (K,)
        omega_z = omega[:, 2]  # (K,)

        # E_base per env (N,)
        E_base = 2.0 * torch.sum(tpos_diff_in_obj * tpos_diff_in_obj, dim=1)
        # multiplier if E_base > 1
        mult = torch.where(E_base > 1.0, E_base, torch.ones_like(E_base))  # (N,)

        # Compute E_dot base per env and kp: -2 * dot(tpos_diff_in_obj, f_kp)
        # (N,3) @ (K,3).T -> (N,K)
        E_dot_base = -2.0 * (tpos_diff_in_obj @ f.t())  # (N,K)
        # apply multiplier per-row
        E_dot_base = E_dot_base * mult.unsqueeze(1)

        # E_total = E_base + tangle_diff**2  (N,)
        E_total = E_base + tangle_diff * tangle_diff

        # E_dot add 2 * tangle_diff * omega_z_kp
        E_dot = E_dot_base + 2.0 * tangle_diff.unsqueeze(1) * omega_z.unsqueeze(0)  # (N,K)

        # normalize by sqrt(tau_z^2 + 1)
        denom = torch.sqrt(tau_z * tau_z + 1.0)  # (K,)
        scores = E_dot / denom.unsqueeze(0)  # (N,K)

        # choose best index per env (argmin over K)
        best_idx = torch.argmin(scores, dim=1).to(torch.long)  # (N,)

        # handle keep_index logic per env: if last_index >=0 and for that kp E_dot <0 and tangle_diff * omega_last <=0
        last = self.last_index  # (N,)
        keep_mask = torch.zeros_like(last, dtype=torch.bool)
        if torch.any(last >= 0):
            valid = last >= 0
            # gather E_dot for last index
            last_idx_clamped = last.clone()
            last_idx_clamped[last_idx_clamped < 0] = 0
            E_dot_last = E_dot.gather(1, last_idx_clamped.unsqueeze(1)).squeeze(1)  # (N,)
            omega_last = omega_z[last_idx_clamped]  # (N,)
            cond = (E_dot_last < 0.0) & (tangle_diff * omega_last <= 0.0)
            # only keep where last was valid
            keep_mask = cond & valid
            # set best_idx to last where keep_mask true
            if keep_mask.any():
                best_idx = torch.where(keep_mask, last, best_idx)

        # update last_index
        self.last_index = best_idx.clone()

        # gather kp pos and normals
        kp_pos = self.key_point_poses[best_idx]  # (N,3)
        kp_pos = kp_pos.clone()
        kp_pos[:, 2] = 0.025
        kp_normal = self.key_point_normals[best_idx]  # (N,3)

        start_point = kp_pos + kp_normal * 0.02

        # movement logic
        targ_z = torch.full((pos.shape[0],), 0.025, device=pos.device)
        speed = torch.full((pos.shape[0],), 0.02, device=pos.device)

        move_dir_xy = start_point[:, :2] - rel_pos_in_obj[:, :2]
        dist_xy = torch.norm(rel_pos_in_obj[:, :2] - start_point[:, :2], dim=1)
        far_mask = dist_xy > 0.005
        targ_z[far_mask] = 0.038
        speed[far_mask] = 0.05

        below_mask = rel_pos_in_obj[:, 2] < 0.035
        zero_xy_mask = far_mask & below_mask
        move_dir_xy[zero_xy_mask, :] = 0.0

        move_dir_z = targ_z - rel_pos_in_obj[:, 2]
        move_direction[:, :2] = move_dir_xy
        move_direction[:, 2] = move_dir_z

        kp_diff = rel_pos_in_obj - kp_pos
        dot = torch.sum(kp_diff * kp_normal, dim=1, keepdim=True)
        ortho_kp_diff = kp_diff - dot * kp_normal
        ortho_norm = torch.norm(ortho_kp_diff, dim=1)
        close_mask = (dot.squeeze(1) > 0.0) & (ortho_norm < 0.005)
        if close_mask.any():
            # where close, override move_direction and set speed
            move_direction[close_mask] = (-kp_normal - ortho_kp_diff * 100.0)[close_mask]
            speed[close_mask] = 0.02

        # normalize and scale by speed
        norms = torch.norm(move_direction, dim=1).clamp_min(1e-8)
        move_dir_unit = move_direction / norms.unsqueeze(1)
        move_delta = move_dir_unit * speed.unsqueeze(1)
        move_delta = R @ move_delta.unsqueeze(-1)
        move_delta = move_delta.squeeze(-1)

        # apply to head_pose
        head_pose = head_pose.clone()
        head_pose[:, :3] = head_pose[:, :3] + move_delta
        head_pose[:, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=head_pose.device)
        return head_pose


class PushTMotionPlanningPolicy(sim_scaling.policy.base_policy.BasePolicy):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.policy_gpu = None


    def get_action(self, obs):
        num_envs = obs["head_pose"].shape[0]
        device = obs["head_pose"].device
        if self.policy_gpu is None or self.policy_gpu.num_envs != num_envs or self.policy_gpu.device != device:
            self.policy_gpu = ControlPolicyGPU(num_envs, device)
            print(f"Initialized ControlPolicyGPU with num_envs={num_envs}, device={device}")
        action = self.policy_gpu.control(obs["head_pose"], obs["t_pose"], obs["targ_pose"])
        return action