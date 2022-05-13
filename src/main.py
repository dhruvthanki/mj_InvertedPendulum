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

Kp = 500
Kd = (math.sqrt(Kp)/2)
data.qpos = np.array([np.deg2rad(0), np.deg2rad(10)])

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    data.ctrl = np.zeros(2) + Kd*(np.zeros(2) - data.qvel) + Kp*(np.zeros(2) - data.qpos)
    # data.ctrl = np.zeros(2) + Kd*(np.zeros(2) - data.sensordata[2:3]) + Kp*(np.zeros(2) - data.sensordata[:2])

    mujoco.mj_step2(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()