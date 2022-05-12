import mujoco
import glfw
import numpy as np
np.set_printoptions(precision=4)
import math

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

model = mujoco.MjModel.from_xml_path('model/InvertedPendulum_v2.xml')
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

scene = mujoco.MjvScene(model, 6000)
camera = mujoco.MjvCamera()
camera.trackbodyid = 1
camera.distance = 2
camera.azimuth = 100
camera.elevation = -20
mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

data.qpos = np.deg2rad(10)

# mujoco.mj_step(model, data)
# ballsite = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ballsite")
# site_pos = data.site_xpos[ballsite]
# print(site_pos)

q_des = np.deg2rad(10)
dq_des = np.deg2rad(10)
ddq_des = np.deg2rad(10)

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    Kp = 150
    Kd = math.sqrt(Kp)/2;
    pd = Kp*(q_des-data.qpos) + Kd*(dq_des-data.qvel)
    b3 = -pd

    data.ctrl = pd
    
    mujoco.mj_step2(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()