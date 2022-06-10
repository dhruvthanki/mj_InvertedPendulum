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

st_jacp = np.zeros((3,model.nv))
st_jacr = np.zeros((3,model.nv))
EndEffector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "EndEffector")
site_pos = data.site_xpos[EndEffector]
mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector);
J_old = st_jacp
M = np.zeros((model.nv,model.nv))
B = np.identity(model.nv)
ctrl_range = model.actuator_ctrlrange[:,1]
u = np.hstack([ctrl_range, 50*np.ones(model.nv)])

Kp = np.array([500, 500, 100, 50, 10, 10])
Kd = (np.sqrt(Kp)/2) + np.array([40, 50, 20, 18, 5, 2])
Kp_osc = 300
Kd_osc = (np.sqrt(Kp_osc)/2) + 20

# data.qpos = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
data.qpos = np.array([np.deg2rad(90), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(-90), np.deg2rad(90), np.deg2rad(90)])
q_des = np.array([np.deg2rad(90), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(-90), np.deg2rad(90), np.deg2rad(90)])
data.qpos = np.zeros(model.nq)

def calc_des(model, data):
    ref = np.array([-0.134,  -0.0994,  1.0795])
    d_ref = np.zeros(3)
    dd_ref = np.zeros(3)
    return ref, d_ref, dd_ref

dummy_state = np.zeros((1,3))
real_states = np.zeros((1,3))
real_states[0,:] = site_pos

ref, d_ref, dd_ref = calc_des(model, data)
des_states = np.zeros((1,3))
des_states[0,:] = ref

time = np.zeros((1,1))
time[0,0] = data.time
dummy_time = np.zeros((1,1))

delta = 1.0e-1

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    site_pos = data.site_xpos[EndEffector]
    # print(site_pos)

    mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector)
    dJac = (st_jacp - J_old)/0.0005
    J_old = st_jacp

    a3 = np.block([np.zeros((3, model.nu)), -st_jacp])
    ref, d_ref, dd_ref = calc_des(model, data)
    ddy_des = dd_ref + Kd_osc*(d_ref-st_jacp@data.qvel) + Kp_osc*(ref-site_pos)
    b3 = dJac@data.qvel - ddy_des

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

    mujoco.mj_step2(model, data)
    
    dummy_state[0,:] = ref
    des_states = np.append(des_states, dummy_state, axis=0)

    dummy_state[0,:] = data.site_xpos[EndEffector]
    real_states = np.append(real_states, dummy_state, axis=0)
    
    dummy_time[0,0] = data.time
    time = np.append(time, dummy_time, axis=0)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

error = 100+(real_states-des_states)
plt.plot(time[:,0], error[:,0], color='r', label='Tracking Error X')
plt.plot(time[:,0], error[:,1], color='b', label='Tracking Error Y')
plt.plot(time[:,0], error[:,2], color='g', label='Tracking Error Z')
plt.xlabel("Time")
plt.ylabel("Magnitude")
# plt.title("Sine and Cosine functions")
# plt.legend()
plt.show()