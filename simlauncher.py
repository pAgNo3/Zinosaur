import argparse
import os
import time

from ament_index_python.packages import get_package_share_directory
import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
class MuJoCoSimLauncher(Node):

  def __init__(self):
    super().__init__("mujoco_sim_launcher")

    pkg_share = get_package_share_directory("DINO_description")
    xml_path = os.path.join(pkg_share, "urdf", "zino.xml")

    self.model = mujoco.MjModel.from_xml_path(xml_path)
    self.data = mujoco.MjData(self.model)
    mujoco.mj_forward(self.model, self.data)

    self.imu_pub = self.create_publisher(Imu, "/imu", 10)
    self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

    raw_actuators = {
        "l_wheel": "left_wheel_motor",
        "r_wheel": "right_wheel_motor",
        "UL_hip": "UL_hip_servo",
        "LL_hip": "LL_hip_servo",
        "UR_hip": "UR_hip_servo",
        "LR_hip": "LR_hip_servo",
    }

    self.actuator_ids = {}
    for key, name in raw_actuators.items():
      aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      if aid == -1:
        self.get_logger().error(f"ACTUATOR NOT FOUND IN XML: '{name}'")
      self.actuator_ids[key] = aid

    self.target_joints = [
        "leftwheel_joint",
        "rightwheel_joint",
        "ULhip_joint",
        "LLhip_joint",
        "URhip_joint",
        "LRhip_joint",
    ]
    self.joint_actuator_map = {
        "leftwheel_joint": "l_wheel",
        "rightwheel_joint": "r_wheel",
        "ULhip_joint": "UL_hip",
        "LLhip_joint": "LL_hip",
        "URhip_joint": "UR_hip",
        "LRhip_joint": "LR_hip",
    }

    self.joint_indices = {}
    for j_name in self.target_joints:
      j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
      if j_id == -1:
        self.get_logger().error(f"JOINT NOT FOUND IN XML: '{j_name}'")
        continue
      self.joint_indices[j_name] = {
          "qpos_idx": self.model.jnt_qposadr[j_id],
          "qvel_idx": self.model.jnt_dofadr[j_id],
          "act_id": self.actuator_ids[self.joint_actuator_map[j_name]],
      }

    self.create_subscription(
        Float64MultiArray, "/robot_cmd", self.command_callback, 10
    )
    self.reset_srv = self.create_service(Trigger, '/zino/reset', self.reset_callback)

    self.get_logger().info("MuJoCo Sim Bridge Ready.")

  def command_callback(self, msg):
    if len(msg.data) == 6:
      for i, key in enumerate(
          ["l_wheel", "r_wheel", "UL_hip", "LL_hip", "UR_hip", "LR_hip"]
      ):
        aid = self.actuator_ids.get(key, -1)
        if aid != -1:
          self.data.ctrl[aid] = float(msg.data[i])

  def publish_imu(self):
    msg = Imu()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "camera_link_1"

    quat = self.data.sensor("camera_imu_quat").data
    (
        msg.orientation.w,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
    ) = map(float, quat)

    gyro = self.data.sensor("camera_imu_gyro").data
    (
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
    ) = map(float, gyro)

    accel = self.data.sensor("camera_imu_accel").data
    (
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z,
    ) = map(float, accel)

    self.imu_pub.publish(msg)

  def publish_joint_states(self):
    msg = JointState()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.name = self.target_joints

    positions, velocities, efforts = [], [], []
    for j_name in self.target_joints:
      idxs = self.joint_indices.get(j_name)
      if idxs and idxs["act_id"] != -1:
        positions.append(float(self.data.qpos[idxs["qpos_idx"]]))
        velocities.append(float(self.data.qvel[idxs["qvel_idx"]]))
        efforts.append(float(self.data.actuator_force[idxs["act_id"]]))
      else:
        positions.append(0.0)
        velocities.append(0.0)
        efforts.append(0.0)

    msg.position, msg.velocity, msg.effort = positions, velocities, efforts
    self.joint_pub.publish(msg)

  def reset_callback(self, request, response):
    mujoco.mj_resetData(self.model, self.data)
    # Set nominal standing base clearance height (0.35m) instead of 0.0m
    self.data.qpos[2] = 0.0
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)
    self.publish_imu()
    self.publish_joint_states()

    response.success = True
    response.message = "Reset complete"
    return response

def main(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--headless",
      type=str,
      default="false",
      help="Run headless without GUI (true/false)",
  )
  parsed_args, _ = parser.parse_known_args()
  is_headless = parsed_args.headless.lower() == "true"

  rclpy.init(args=args)
  sim = MuJoCoSimLauncher()
  executor = SingleThreadedExecutor()
  executor.add_node(sim)

  try:
    if is_headless:
      frame_skip = 10
      dt = sim.model.opt.timestep
      next_time = time.perf_counter()
      while rclpy.ok():

        executor.spin_once(timeout_sec=0.0)
        for _ in range(frame_skip):
            mujoco.mj_step(sim.model, sim.data)
        
        sim.publish_imu()
        sim.publish_joint_states()
        
        next_time += dt
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Resynchronize if simulation falls behind wall-clock
            next_time = time.perf_counter()
    else:
      with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running() and rclpy.ok():
          step_start = time.time()
          mujoco.mj_step(sim.model, sim.data)
          sim.publish_imu()
          sim.publish_joint_states()
          executor.spin_once(timeout_sec=0.0)
          viewer.sync()

          elapsed = time.time() - step_start
#          if elapsed < 0.02:
#            time.sleep(0.02 - elapsed)
  except KeyboardInterrupt:
    pass
  finally:
    sim.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
