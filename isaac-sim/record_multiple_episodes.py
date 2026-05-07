import numpy as np
from datetime import datetime
import os
import shutil
import math

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLightning",
    "physics_dt": 1/60
})

from isaacsim.core.api import World

from envs.arena import Arena
from envs.robot import Create3Robot

world = World()
print("World generated")

arena = Arena(world)
arena.build()
print("Arena built")

robot = Create3Robot(world)
robot.load()
print("Robot created")

world.reset()
robot.initialize()
print("Physics initialized")

for i in range(60):
    world.step(render=True)

import omni.replicator.core as rep
import carb
import cv2

# rep.orchestrator.set_capture_on_play(False)

camera = rep.create.camera(
    position=(0.0, 0.0, 2.0),
    rotation=(-180, -90, 0),
    focal_length=12.0,
    clipping_range=(0.01, 100)
)
print("Camera set up")

render_product = rep.create.render_product(camera, resolution=(1024, 1024))

carb.settings.get_settings().set("/exts/omni.replicator.core/maxAssetLoadingTime", 10.0)
carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)

output_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records"
os.makedirs(output_path, exist_ok=True)

images_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records/images/"
os.makedirs(images_path, exist_ok=True)

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=images_path, rgb=True)
writer.attach([render_product])

images = []

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = "episode_" + timestamp + ".mp4"
video_path = os.path.join(output_path, video_name)

video = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'DIVX'),
    60,
    (1024, 1024)
)

for ep in range(1, 11):

    world.reset()
    robot.initialize()

    spawn_x = np.random.uniform(-0.7, 0.7)
    spawn_y = np.random.uniform(-0.7, 0.7)
    spawn_yaw = np.random.uniform(-math.pi, math.pi)

    robot.teleport(position=np.array([spawn_x, spawn_y, 0.0]), yaw=spawn_yaw)
    
    for i in range(60):
        world.step(render=True)

    spawn_x = round(spawn_x, 2)
    spawn_y = round(spawn_y, 2)
    print(f"Ep: {ep}, spawned robot in ({spawn_x}, {spawn_y})")

    linear_vel = np.random.uniform(-0.3, 0.5)
    angular_vel = np.random.uniform(-0.5, 0.5)

    linear_vel = round(linear_vel, 2)
    angular_vel = round(angular_vel, 2)
    print(f"Ep: {ep}, command: ({linear_vel}, {angular_vel})")

    for step in range(150):

        robot.apply_action(command=np.array([linear_vel, angular_vel]))
        world.step(render=True)

        if step%30 == 0:
            x, y, z = robot.get_state()["position"]
            x = round(x, 2)
            y = round(y, 2)
            z = round(z, 2)
            distance = round(float(np.linalg.norm(np.array([x, y]) - np.zeros(2))), 2)
            print(f"Ep: {ep}, step : {step}, coords = ({x}, {y}, {z}), distance = {distance}")

    print(len([img for img in os.listdir(images_path)]))

rep.orchestrator.step(rt_subframes=8, delta_time=0.0)

for _ in range(60):
        world.step(render=True)

images = [img for img in os.listdir(images_path)]
images.sort()

for image in images:
    video.write(cv2.imread(os.path.join(images_path, image)))

video.release()
print("Video succesfully generated !")

shutil.rmtree(images_path)

print(f"Command to use to get the video : scp g.portron@10.163.11.19:{video_path} Téléchargements/")

app.close()
