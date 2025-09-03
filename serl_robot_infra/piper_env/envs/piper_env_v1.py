import requests
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R


K_cartesian = np.diag([1.2, 1., 1., 0.1, 0.1, 0.1])
K_cartesian *= 90
D_cartesian = np.diag([1.5, 1.3, 1.3, 0.1, 0.1, 0.1])
D_cartesian *= 5

k1 = 4.
k2 = 0.87
sign = np.array([1, 1, 1, 1, 1, 1])

url = "http://100.72.16.108:5001/"
start_pose = np.array([0.25705429, -0.01145554, 0.35798343, -2.08731568, -1.40663244, 0.47628889])

class piper_env():
        def __init__(self):
                self.kp_lift = 1 * np.array([0, 0, 0, 1, 0, 0])
                self.kd_lift = 0.2 * np.array([0.4, 0.8, 1., 1.1, 1.5, 1.0])
                self.k_joint = 1. *np.array([2., 5., 4, 1, 0.5, 0.5])

        def update_curr_full_state(self, pose_desire):
                data = {"pose_desire": pose_desire.tolist()}
                ps = requests.post(url + "getfullstate", json=data).json()
                q = np.array(ps["q"])
                dq = np.array(ps["dq"])
                tau = np.array(ps["tau"])
                J = np.array(ps["J"])
                g = np.array(ps["g"])
                coriolis = np.array(ps["coriolis"])
                bTe = np.array(ps["bTe"])
                pose_error = np.array(ps["pose_error"])
                # print(q, end='')
                # print(J, end="\r", flush=True)
                # print(q[2])

                return q, dq, tau, J, g, coriolis, bTe, pose_error
        
        def check_is_takenover(self):
                ps = requests.post(url + "get_takeover").json()
                is_taken = ps["is_taken"]
                return is_taken

        def send_PDmove_command(slef, pos):
                arr = np.array(pos).astype(np.float32)
                data = {"arr": arr.tolist()}
                requests.post(url + "PDmove", json=data)

        def send_MITmove_command(self, q_d, kp_lift, kd_lift, tau):
                q_d = np.array(q_d).astype(np.float32)
                arr = np.array(tau).astype(np.float32)
                kp_lift = np.array(kp_lift).astype(np.float32)
                kd_lift = np.array(kd_lift).astype(np.float32)
                data = {"arr": arr.tolist(),
                        "kp_lift": kp_lift.tolist(),
                        "kd_lift": kd_lift.tolist(),
                        "q_d": q_d.tolist()}
                requests.post(url + "MITmove", json=data)

        def send_MITmove_delta_command(self, tau_apply_last, start_pose = start_pose, delta_pose = np.array([0, 0, 0, 0, 0, 0])):
                pose_desire = start_pose + delta_pose
                tau_apply = np.array([0., 0., 0., 0., 0., 0.])
                q, dq, tau_last, J, g, coriolis, bTe, pose_error = self.update_curr_full_state(pose_desire)
                v = J @ dq

                tau_task = J.T @ (K_cartesian @ pose_error + D_cartesian @ (- v))
                tau_task = tau_task * self.k_joint
                tau_ori = tau_task + g + coriolis
                ori = tau_ori
                ori[:3] = ori[:3] / k1
                ori[3:] = ori[3:] / k2
                tau_new = ori * sign

                # tau_apply_last = tau_apply

                tau_diff = tau_new - tau_apply_last
                tau_diff = np.clip(tau_diff, -0.1, 0.1)
                tau_apply = tau_diff + tau_apply_last
                tau_apply = np.clip(tau_apply, -2, 2)
                # print(tau_apply)
                q_d = [0, 0, 0, 0, 0, 0]
                q_d[1] *= -1
                self.send_MITmove_command(q_d, self.kp_lift, self.kd_lift, tau_apply)

                print(pose_error)

                return tau_apply

        def send_MITmvoe_delta_q_command(self, start_pose = start_pose, delta_pose = np.array([0, 0, 0, 0, 0, 0])):
                pose_desire = start_pose + delta_pose
                damping=1e-3
                q, dq, tau_last, J, g, coriolis, bTe, pose_error = self.update_curr_full_state(pose_desire)
                JJt = J @ J.T
                delta_q = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(6), pose_error)

                q_new = q + delta_q
                q_new[1] *= -1
                
                tau_comp = g
                ori = tau_comp
                ori[:3] = ori[:3] / k1
                ori[3:] = ori[3:] / k2
                tau_ff = ori * sign

                self.send_MITmove_command(q_new, self.kp_lift, self.kd_lift, tau_ff)



# ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
# ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])


pos = [0, 0.3, -2.5, 0, 0, 0, 0.]




env = piper_env()

iter = 0
HZ = 180
delta_pose = np.array([0., 0., 0., 0., 0., 0.])

tau_apply = np.array([0., 0., 0., 0., 0., 0.])
t_start = time.time()
while True:
        back = False
        tau_last = tau_apply
        if not env.check_is_takenover():
                tau_apply = env.send_MITmove_delta_command(tau_last, start_pose, delta_pose)
        iter+=1

        if iter > 6000:
                back = True

        if iter > 2000 and iter % 18 == 0:
                if not back:
                        if delta_pose[0]<0.15:
                                delta_pose[0] += 0.005
                        elif delta_pose[1]<0.15:
                                delta_pose[1] += 0.005
                        elif delta_pose[2]>-0.15:
                                delta_pose[2] += -0.005
                else:
                        if delta_pose[2]<0:
                                delta_pose[2] += 0.005
                        elif delta_pose[1]>0:
                                delta_pose[1] += -0.005
                        elif delta_pose[0]>0:
                                delta_pose[0] += -0.005
                        
        

        
                
        # pose_desire = start_pose + delta_pose
        # q, dq, tau_last, J, g, coriolis, bTe, pose_error = env.update_curr_full_state(start_pose)
        

        # v = J @ dq
        # v_diff = v_new - v
        # v_diff = np.clip(v_diff, -0.02, 0.02)
        # v = v_diff + v

        # tau_task = J.T @ (K_cartesian @ pose_error + D_cartesian @ (- v))
        # tau_ori = tau_task + g

        # dq = np.round(dq, 4)
        # print(g)
        # print(dq)
        # pos = bTe[:3, 3] 
        # rot = R.from_matrix(bTe[:3, :3])
        # rpy = rot.as_euler('xyz') # 世界坐标系按xyz外旋得link8坐标系
        # print(pos, end=" ")
        # print(rpy)
        # print(pose_error)



        # print(tau_last)
        # print(g)
        # print(dq)
        
        # ori = np.array([ 0.00000000e+00, 1.14426254e+01, 4.38430614e+00, 9.30506873e-03, 8.39613637e-01, -3.51747428e-04])
        # ori = tau_ori
        # ori[:3] = ori[:3] / k1
        # ori[3:] = ori[3:] / k2
        # tau_new = ori * sign
        
        # alpha = 1
        # tau_apply = alpha*tau_new + (1-alpha)*tau_apply

        # tau_diff = tau_new - tau_apply_last
        # tau_diff = np.clip(tau_diff, -0.1, 0.1)
        # tau_apply = tau_diff + tau_apply_last
        # tau_apply = np.clip(tau_apply, -1, 1)
        # print(tau_apply)

        

        # print(g)
        # g_ref = g
        # g_ref[:3] = g_ref[:3] / k1
        # g_ref[3:] = g_ref[3:] / k2
        # g_ref = g_ref * sign
        # print(g_ref)
        # tau_apply = np.array([0, -1.16, 0.384, 0, 0.5, 0])
        # print(tau_apply)
        # if tau_apply > 
        
        # print(kp_lift)
        # q_d = np.array(pos)
        # q_d = q
        # q_d[1]*= -1
        # env.send_MITmove_command(q_d, kp_lift, kd_lift, tau_apply)

        # if iter>9000:
        #         break

        # env.send_PDmove_command(pos)

# t_1 = time.time()
# print(5000/(t_1 - t_start))

# t1 = time.time()
# p = 0
# while True:
#         update_curr_full_state()
#         time.sleep(0.1)
#         # os.system("clear")
#         p+=1
        
#         if p>1000:
#                 break

# t2 = time.time()
# print(1001/(t2-t1))


