import mujoco
import glfw
import numpy as np
np.set_printoptions(precision=4)
import math
import matplotlib.pyplot as plt
import osqp
from scipy import sparse

def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                       title='Demo', monitor=None,
                                       share=None)
    glfw.make_context_current(window)
    return window

window = init_window(2400, 1800)
width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

model = mujoco.MjModel.from_xml_path('model/ur5/ur5.xml')
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

scene = mujoco.MjvScene(model, 6000)
camera = mujoco.MjvCamera()
camera.trackbodyid = 2
camera.distance = 3
camera.azimuth = 90
camera.elevation = -50
mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

q_des = np.array([np.deg2rad(0), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
# data.qpos = np.zeros(model.nq)
data.qpos = np.array([ 1.479,  -2.2861,  0.935,  -1.9355,  4.6084, -6.0808])
mujoco.mj_forward(model,data)

st_jacp = np.zeros((3,model.nv))
st_jacr = np.zeros((3,model.nv))
EndEffector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "EndEffector")
site_pos = data.site_xpos[EndEffector]
mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector);
# print(st_jacp)
J_old = st_jacp
M = np.zeros((model.nv,model.nv))
B = np.identity(model.nv)
ctrl_range = model.actuator_ctrlrange[:,1]
u = np.hstack([ctrl_range, 50*np.ones(model.nv)])

Kp = np.array([500, 500, 100, 50, 10, 10])
Kd = (np.sqrt(Kp)/2) + np.array([40, 50, 20, 18, 5, 2])
Kp_osc = 400
Kd_osc = (np.sqrt(Kp_osc)/2) + 40

def calc_des(model, data):
    A = 0.20
    omega = 1#2*np.pi*1
    # ref = np.array([-0.134,  -0.0994,  1.0795-1.5*A+A])
    # d_ref = np.zeros(3)
    # dd_ref = np.zeros(3)
    ref = np.array([-0.134,  -0.0994+A*np.sin(omega*data.time),  1.0795-1.5*A+A*np.cos(omega*data.time)])
    d_ref = np.array([0,  A*np.cos(data.time),  -A*np.sin(data.time)])
    dd_ref = np.array([0,  -A*np.sin(data.time),  -A*np.cos(data.time)])
    ref_ori = np.array([0.707, 0.707, 0, 0])
    # if data.time < 1:
    #     print(np.block([[data.qpos, data.qvel]]))
    return ref, d_ref, dd_ref, ref_ori

real_states = np.reshape(site_pos, (1, 3))
ref, d_ref, dd_ref, ref_ori = calc_des(model, data)
A = 0.20
data.qvel = np.linalg.pinv(st_jacp)@np.array([0, A, 0])
# print(data.qvel)
des_states = np.reshape(ref, (1, 3))
time = np.reshape(data.time, (1, 1))
input = np.zeros((1,6))

delta = 1.0e-7

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    site_pos = data.site_xpos[EndEffector]
    site_ori = data.site_xmat[EndEffector]
    # print(site_pos)

    mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector)
    dJac = (st_jacp - J_old)/0.0005
    J_old = st_jacp

    a3 = np.block([np.zeros((3, model.nu)), -st_jacp])
    ref, d_ref, dd_ref, ref_ori = calc_des(model, data)
    ddy_des_pos = dd_ref + Kd_osc*(d_ref-st_jacp@data.qvel) + Kp_osc*(ref-site_pos)
    # ddy_des_ori = np.zeros((1,4)) + Kd_osc*(np.zeros((1,4))-st_jacr@data.qvel) + Kp_osc*(ref-site_pos)
    b3 = dJac@data.qvel - ddy_des_pos

    mujoco.mj_fullM(model, M, data.qM)
    Bias = data.qfrc_bias

    Aeq = np.block([[-B, M]])
    beq = -Bias

    # OSQP Setup
    # https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
    Q = (a3.transpose()).dot(a3) + delta*np.eye(12)
    q = -(a3.transpose()).dot(b3)
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

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # print(np.block([[data.qpos, data.qvel]]))

glfw.terminate()

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