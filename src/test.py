import numpy as np
np.set_printoptions(precision=4)
import math
import matplotlib.pyplot as plt
import osqp
from scipy import sparse
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
# import pyquaternion as quat
from pyquaternion import Quaternion
import threading

model = mujoco.MjModel.from_xml_path('/home/dhruv/Documents/GitHub/mj_InvertedPendulum2/model/ur5/ur5.xml')
data = mujoco.MjData(model)

q_des = np.array([np.deg2rad(0), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
# data.qpos = np.zeros(model.nq)
# data.qpos = np.array([ 1.479,  -2.2861,  0.935,  -1.9355,  4.6084, -6.0808])
data.qpos = np.array([ -0.0196,  -2.38,  1.71,  -2.47,  4.73, -4.71])
mujoco.mj_forward(model,data)

EndEffector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "EndEffector")
site_pos = data.site_xpos[EndEffector]
site_ori = data.site_xmat[EndEffector]
site_ori = np.reshape(site_ori, (3,3))
r = R.from_matrix(site_ori)
curr_site_quat = r.as_quat() # [x, y, z, w]

st_jacp = np.zeros((3,model.nv))
st_jacr = np.zeros((3,model.nv))
mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector);
# print(st_jacp)
J_old = st_jacp
J_old_r = st_jacr
M = np.zeros((model.nv,model.nv))
B = np.identity(model.nv)
ctrl_range = model.actuator_ctrlrange[:,1]
u = np.hstack([ctrl_range, 50*np.ones(model.nv)])

# Kp = np.array([500, 500, 100, 50, 10, 10])
# Kd = (np.sqrt(Kp)/2) + np.array([40, 50, 20, 18, 5, 2])
Kp_osc = 400
Kd_osc = (np.sqrt(Kp_osc)/2) + 40

Kp_osc_r = 200
Kd_osc_r = (np.sqrt(Kp_osc_r)/2) + 40

Kp_osc_f = 200
 
def handle_quat_sign_flip(new_quat, curr_quat):
    if np.dot(curr_quat, new_quat) < 0:
        new_quat = -new_quat
    return new_quat

def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices
    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix
    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error

def calc_des(timeT):
    A = 0.20
    omega = 1#2*np.pi*1
    # ref = np.array([-0.134,  -0.0994,  1.0795-1.5*A+A])
    # d_ref = np.zeros(3)
    # dd_ref = np.zeros(3)
    # ref = np.array([-0.134,  -0.0994+A*np.sin(omega*timeT),  1.0795-1.5*A+A*np.cos(omega*timeT)])
    ref = np.array([0.5,  -0.0994+A*np.sin(omega*timeT),  1.0795-1.5*A+A*np.cos(omega*timeT)])
    # ref = np.array([0.4,  A,  0.8])
    d_ref = np.array([0,  A*np.cos(timeT),  -A*np.sin(timeT)])
    dd_ref = np.array([0,  -A*np.sin(timeT),  -A*np.cos(timeT)])
    r = R.from_euler('XYZ', [0,90,0], degrees=True)
    # ref_ori = r.as_quat() # [x, y, z, w]
    ref_ori = r.as_matrix()
    # if timeT < 1:
    #     print(np.block([[data.qpos, data.qvel]]))
    force_ref = 0.1
    return ref, d_ref, dd_ref, ref_ori, force_ref

real_states = np.reshape(site_pos, (1, 3))
ref, d_ref, dd_ref, ref_ori, force_ref = calc_des(data.time)
A = 0.20
data.qvel = np.linalg.pinv(st_jacp)@np.array([0, A, 0])
# print(data.qvel)
des_states = np.reshape(ref, (1, 3))
time = np.reshape(data.time, (1, 1))
input = np.zeros((1,6))

delta = 1.0e-4

viz_thread = viewer.launch_passive(model, data)
while(viz_thread.is_alive()):
    mujoco.mj_step1(model, data)

    # print(site_pos)
    # r = R.from_matrix(site_ori)
    # new_site_quat = r.as_quat() # [x, y, z, w]
    # # curr_site_quat = handle_quat_sign_flip(new_site_quat, curr_site_quat)
    # curr_site_quat = new_site_quat

    mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector)

    ref, d_ref, dd_ref, ref_ori, force_ref = calc_des(data.time)
    
    # if not (data.sensordata[0] == 0):
    print(data.sensordata[0])

    # Force Control direction
    a1 = np.block([np.zeros((1, model.nu+6)), np.ones((1,1))])
    # force_diff  = force_ref - data.sensordata[0]
    b1 = -force_ref 

    # Motion control direction
    a2 = np.block([np.zeros((3, model.nu)), -st_jacr, np.zeros((3,1))])
    site_ori = data.site_xmat[EndEffector]
    site_ori = np.reshape(site_ori, (3,3))
    diff_orientation_err = orientation_error(ref_ori, site_ori)
    ddy_des_ori = np.zeros((diff_orientation_err.shape[0])) + Kd_osc_r*(np.zeros((diff_orientation_err.shape[0]))-st_jacr@data.qvel) + Kp_osc_r*(diff_orientation_err)
    dJac_r = (st_jacr - J_old_r)/0.0005
    J_old_r = st_jacr
    b2 = dJac_r@data.qvel - ddy_des_ori
    
    dim=3
    a3 = np.block([np.zeros((dim, model.nu)), -st_jacp[-dim:, :], np.zeros((dim,1))])
    site_pos = data.site_xpos[EndEffector]
    ddy_des_pos = dd_ref[-dim:] + Kd_osc*(d_ref[-dim:]-st_jacp[-dim:, :]@data.qvel) + Kp_osc*(ref[-dim:]-site_pos[-dim:])
    # ddy_des_ori = np.zeros((1,4)) + Kd_osc*(np.zeros((1,4))-st_jacr@data.qvel) + Kp_osc*(ref-site_pos)
    dJac = (st_jacp - J_old)/0.0005
    J_old = st_jacp
    b3 = dJac[-dim:, :]@data.qvel - ddy_des_pos

    mujoco.mj_fullM(model, M, data.qM)
    Bias = data.qfrc_bias

    # Equality Constraints
    Aeq = np.block([[-B, M, -st_jacp.T[:, 0].reshape((6,1)) ]])
    beq = -Bias

    # OSQP Setup
    # https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
    Q = 2*(a3.transpose()).dot(a3) + 50*(a2.transpose()).dot(a2) + 50*(a1.transpose()).dot(a1) + delta*np.eye(13)
    q = -2*(a3.transpose()).dot(b3) - 50*(a2.transpose()).dot(b2) - 50*(a1.transpose()).dot(b1)
    P = sparse.csc_matrix(Q)
    A = sparse.csc_matrix(Aeq)

    # OSQP
    prob = osqp.OSQP()
    prob.setup(P, q, A, beq, beq, verbose=False)
    try:
        res = prob.solve()
    except:
        pass
    data.ctrl = res.x[:model.nv]
    # data.ctrl = np.zeros(model.nv) + Kd*(np.zeros(model.nv) - data.qvel) + Kp*(q_des - data.qpos)

    mujoco.mj_step2(model, data)

    des_states = np.append(des_states, np.reshape(ref, (1, 3)), axis=0)
    real_states = np.append(real_states, np.reshape(site_pos, (1, 3)), axis=0)
    time = np.append(time, np.reshape(data.time, (1, 1)), axis=0)
    input = np.append(input, np.reshape(data.ctrl, (1, 6)), axis=0)

plt.figure()

plt.subplot(131)
plt.plot(real_states[:,1], real_states[:,2], color='b', label='Tracked')
plt.plot(des_states[:,1], des_states[:,2], color='g', label='Desired')
plt.xlim(-0.4, 0.2)
plt.ylim(0.5, 1)
plt.xlabel("Y-axis (m)")
plt.ylabel("Z-axis (m)")
plt.axis("Equal")
plt.title("EE Traj (YZ-plane)")
plt.legend()

plt.subplot(132)
error = 1000*(real_states-des_states)
plt.plot(time[:,0], error[:,0], color='r', label='X')
plt.plot(time[:,0], error[:,1], color='b', label='Y')
plt.plot(time[:,0], error[:,2], color='g', label='Z')
plt.xlabel("Time (s)")
plt.ylabel("Magnitude (mm)")
plt.title("Trajectory Tracking Error")
plt.legend()

plt.subplot(133)
plt.plot(time[:,0], input[:,0], label='1')
plt.plot(time[:,0], input[:,1], label='2')
plt.plot(time[:,0], input[:,2], label='3')
plt.plot(time[:,0], input[:,3], label='4')
plt.plot(time[:,0], input[:,4], label='5')
plt.plot(time[:,0], input[:,5], label='6')
plt.xlabel("Time (s)")
plt.ylabel("Torque")
plt.title("Joint Input")
plt.legend()

# set the spacing between subplots
plt.subplots_adjust(wspace=0.55)
plt.suptitle('Dynamic Simulation')
plt.savefig('src/ur5.svg', format='svg')
plt.show()