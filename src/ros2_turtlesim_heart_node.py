"""ROS 2 / turtlesim version of the same learned policy.

Usage (ROS 2 Humble example):
  ros2 run turtlesim turtlesim_node
  python3 src/ros2_turtlesim_heart_node.py

The node subscribes /turtle1/pose and publishes /turtle1/cmd_vel.
Place output/heart_policy.pt next to this project before running.
"""
from pathlib import Path
import sys
import numpy as np
import torch

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from turtlesim.msg import Pose
except ImportError as e:
    raise SystemExit('ROS 2 Python packages are required. Run inside a ROS 2 environment.') from e

sys.path.append(str(Path(__file__).resolve().parent))
from common import load_policy, heart_waypoints, policy_state

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'output' / 'heart_policy.pt'

class HeartPolicyNode(Node):
    def __init__(self):
        super().__init__('ai_heart_policy')
        self.policy = load_policy(MODEL)
        raw = heart_waypoints(n=75, scale=0.18)
        # Map about [-3,3] into turtlesim coordinates [1,10].
        self.points = np.empty_like(raw)
        self.points[:,0] = 5.5 + raw[:,0] * 1.25
        self.points[:,1] = 5.6 + raw[:,1] * 1.05
        self.idx = 0
        self.pose = None
        self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.create_subscription(Pose, '/turtle1/pose', self.pose_cb, 10)
        self.create_timer(0.05, self.step)
        self.get_logger().info('AI heart policy started')

    def pose_cb(self, msg):
        self.pose = msg

    def step(self):
        if self.pose is None or self.idx >= len(self.points):
            return
        st = policy_state(self.pose.x, self.pose.y, self.pose.theta, self.points[self.idx])
        with torch.no_grad():
            v, w = self.policy(torch.from_numpy(st).unsqueeze(0))[0].numpy()
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub.publish(cmd)
        if st[2] < 0.13:
            self.idx += 1
            if self.idx >= len(self.points):
                self.pub.publish(Twist())
                self.get_logger().info('Heart completed')


def main():
    rclpy.init()
    node = HeartPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
