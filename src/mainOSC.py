import mujoco
import glfw
import numpy as np
np.set_printoptions(precision=4)
import osqp
from scipy import sparse

# import imageio
# create a video writer with imageio
# writer = imageio.get_writer("video.mp4", fps=20)

def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                       title='OSC', monitor=None,
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

data.qpos = np.array([np.deg2rad(45), np.deg2rad(45)])
mujoco.mj_forward(model, data)

EndEffector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "EndEffector")
site_pos = data.site_xpos[EndEffector]
ref = np.array([0, 1])
d_ref = np.zeros(2)
dd_ref = np.zeros(2)
gear_ratio = model.actuator_gear
gear_ratio = gear_ratio[:,0]
ctrl_range = model.actuator_ctrlrange[:,1]
indices = np.array([0,2])
st_jacp = np.zeros((3,2))
st_jacr = np.zeros((3,2))
mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector);
J_old = st_jacp[indices,:]
Kp = 1000
Kd = (np.sqrt(Kp)/2) + 60
M = np.zeros((2,2))
B = np.identity(2)
u = np.hstack([ctrl_range, 50*np.ones(2)])

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    site_pos = data.site_xpos[EndEffector]
    xz_pos = site_pos[indices]

    mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector)
    Jac = st_jacp[indices,:]
    dJac = (Jac - J_old)/0.0005
    J_old = Jac

    a3 = np.block([np.zeros((2, 2)), -Jac])
    ddy_des = dd_ref + Kd*(d_ref-Jac@data.qvel) + Kp*(ref-xz_pos)
    b3 = dJac@data.qvel - ddy_des

    mujoco.mj_fullM(model, M, data.qM)
    Bias = data.qfrc_bias

    Aeq = np.block([[-B, M]])
    beq = -Bias

    # OSQP Setup
    # https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
    Q = (a3.transpose()).dot(a3)
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
    data.ctrl = res.x[:2]
    # print(data.ctrl)

    mujoco.mj_step2(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # writer.append_data(frame)
glfw.terminate()
# writer.close()