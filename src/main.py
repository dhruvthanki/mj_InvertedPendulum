import mujoco
import glfw
import numpy as np
np.set_printoptions(precision=4)
import math
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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

sensor_states = np.zeros((1,4))
dummy_state = np.zeros((1,4))
sensor_states[0,:] = (data.sensordata)
time = np.zeros((1,1))
time[0,0] = data.time
dummy_time = np.zeros((1,1))

# my_filter = KalmanFilter(dim_x=2, dim_z=1)
# my_filter.x = np.array([[2.],
#                 [0.]])       # initial state (location and velocity)

# my_filter.F = np.array([[1.,1.],
#                 [0.,1.]])    # state transition matrix

# my_filter.H = np.array([[1.,0.]])    # Measurement function
# my_filter.P *= 1000.                 # covariance matrix
# my_filter.R = 5                      # state uncertainty
# my_filter.Q = Q_discrete_white_noise(2, 0.0005, .1) # process uncertainty

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    data.ctrl = np.zeros(2) + Kd*(np.zeros(2) - data.qvel) + Kp*(np.zeros(2) - data.qpos)
    # data.ctrl = np.zeros(2) + Kd*(np.zeros(2) - data.sensordata[2:3]) + Kp*(np.zeros(2) - data.sensordata[:2])

    mujoco.mj_step2(model, data)

    # my_filter.predict()
    # my_filter.update(get_some_measurement())
    # x = my_filter.x

    dummy_state[0,:] = (data.sensordata)
    sensor_states = np.append(sensor_states, dummy_state, axis=0)
    dummy_time[0,0] = data.time
    time = np.append(time, dummy_time, axis=0)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

plt.plot(time[:,0], sensor_states[:,0])
plt.show()