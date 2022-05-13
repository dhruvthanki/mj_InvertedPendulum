import mujoco
import glfw
import numpy as np
np.set_printoptions(precision=4)
import math
from dataclasses import dataclass
import cvxpy as cp

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

model = mujoco.MjModel.from_xml_path('model/DoublePendulum.xml')
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

scene = mujoco.MjvScene(model, 6000)
camera = mujoco.MjvCamera()
camera.trackbodyid = 2
camera.distance = 3
camera.azimuth = 90
camera.elevation = -20
mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

data.qpos = np.array([np.deg2rad(10), np.deg2rad(10)])
EndEffector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "EndEffector")
q_des = np.array([0, 0.9])
dq_des = np.zeros(2)
ddq_des = np.zeros(2)
gear_ratio = model.actuator_gear
gear_ratio = gear_ratio[:,0]
J_old = np.zeros((2,2))
ctrl_range = model.actuator_ctrlrange[:,1]
init_guess = np.zeros((2))

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    site_pos = data.site_xpos[EndEffector]
    xz_pos = np.array([site_pos[0], site_pos[2]])
    # print(xz_pos)

    st_jacp = np.zeros((3,2))
    st_jacr = np.zeros((3,2))
    mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector);
    indices = np.array([0,2])
    Jac = st_jacp[indices,:]
    # print(Jac)
    dJac = (Jac - J_old)/0.0005
    J_old = Jac

    Kp = 500
    Kd = (math.sqrt(Kp)/2)

    a3 = np.block([np.zeros((2, 2)), -np.identity(2)])
    pd = Kp*(q_des-Jac@xz_pos) + Kd*(dq_des-dJac@data.qvel)
    b3 = -ddq_des - pd

    M = np.zeros((2,2))
    mujoco.mj_fullM(model, M, data.qM)
    Bias = data.qfrc_bias

    Q = (a3.transpose()).dot(a3)
    q = (-a3.transpose()).dot(b3)
    Aeq = np.block([[np.identity(2),-M]])
    beq = Bias

    x = cp.Variable(4)
    objective = cp.Minimize(cp.sum_squares(a3 @ x - b3))
    u = np.hstack([ctrl_range, 500*np.ones(2)])
    constraints = [Aeq @ x == beq, x >= -u, x <= u]
    prob = cp.Problem(objective, constraints)
    
    try:
        result = prob.solve(verbose=True)
    except:
        pass
    data.ctrl = x.value[:2]

    mujoco.mj_step2(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()