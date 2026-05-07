import numpy as np
from datetime import datetime
import os
import shutil

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLightning",
    "physics_dt": 1/30
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

robot.teleport(position=np.array([0.0, 0.0, 0.0]), yaw=0.0)

for i in range(60):
    world.step(render=True)

import omni.replicator.core as rep
import carb
import cv2

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

images_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records/images"
os.makedirs(images_path, exist_ok=True)

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=images_path, rgb=True)
writer.attach([render_product])

for step in range(150):

    robot.apply_action(command=np.array([0.2, 0.5]))
    world.step(render=True)

    if step%30 == 0:
        x, y = robot.get_state()["position"][:2]
        x = round(x, 2)
        y = round(y, 2)
        distance = round(float(np.linalg.norm(np.array([x, y]) - np.zeros(2))), 2)
        print(f"step : {step}, coords = ({x}, {y}), distance = {distance}")

rep.orchestrator.step(rt_subframes=4, delta_time=0.0)

for _ in range(5):
    world.step(render=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = "episode_" + timestamp + ".mp4"
video_path = os.path.join(output_path, video_name)

images = [img for img in os.listdir(images_path)]
images.sort()

video = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'DIVX'),
    30,
    (1024, 1024)
)

for image in images:
    video.write(cv2.imread(os.path.join(images_path, image)))

video.release()

print("Video succesfully generated !")

shutil.rmtree(images_path)
print("Images directory cleared")

print(f"Command to use to get the video : scp g.portron@10.163.11.19:{video_path} Téléchargements/")

app.close()
