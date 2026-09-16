import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
import math
import numpy as np
from sensor_msgs.msg import JointState

def solve_leg_ik(x: float, y: float, a=142.5, b=315.0, c=324.0):
    E1 = -2 * a * x
    E4 = 2 * a * (-x + c)
    F1 = -2 * a * y
    F4 = -2 * a * y
    G1 = (a**2) - (b**2) + (x**2) + (y**2)
    G4 = (c**2) + (a**2) - (b**2) + (x**2) + (y**2) - 2 * c * x

    disc1 = np.maximum(0.0, E1**2 + F1**2 - G1**2)
    disc4 = np.maximum(0.0, E4**2 + F4**2 - G4**2)
    denom1 = (G1 - E1)
    denom4 = (G4 - E4)
#    denom1 = np.where(np.abs(G1 - E1) < 1e-6, 1e-6, G1 - E1)
#    denom4 = np.where(np.abs(G4 - E4) < 1e-6, 1e-6, G4 - E4)

    t1 = 2 * np.arctan((-F1 + np.sqrt(disc1)) / denom1)
    t4 = 2 * np.arctan((-F4 - np.sqrt(disc4)) / denom4)

    return t1, t4
class IKtester(Node):
    def __init__(self):
        super().__init__('pitch_balance_node')

        self.xL=0.6
        self.yL=350.0
        self.xR=0.6
        self.yR=350.0
        # 2. Interfaces
        self.jointte = self.create_subscription(JointState, '/joint_states', self.jointcallback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/robot_cmd', 10)
        self.timer = self.create_timer(0.02, self.Set_to_init)
        self.get_logger().info("IK Tester Node started. Publishing at 50 Hz...")

    def _Normalizer(self, xp, y):
        R = 457.5 - 25.0
        xmin = 324.0 - np.sqrt(np.maximum(0.0, (R**2) - (y**2)))
        xmax = np.sqrt(np.maximum(0.0, (R**2) - (y**2)))

        return xmin + xp*(xmax-xmin)


    def Set_to_init(self):
        Xl = self._Normalizer(self.xL, self.yL)
        Xr = self._Normalizer(self.xR, self.yL)
        t1_L, t4_L = solve_leg_ik(Xl, self.yL)
        t1_R, t4_R = solve_leg_ik(Xr, self.yR)
        
        TL1 = -t1_L + (3.14 / 2)
        TL4 = t4_L - 1.57
        TR1 = -t1_R + (3.14 / 2)
        TR4 = t4_R - 1.57
        
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [0.0, 0.0, float(TL4), float(TL1), float(TR1), float(TR4)]
        self.cmd_pub.publish(cmd_msg)
        self.get_logger().info(
            f"FK Computed Position -> Left Leg (X: {Xl:.2f} mm, Y: {self.yL:.2f} mm)"
        )
        
    def jointcallback(self, msg):
        required_joints = [
            "ULhip_joint",
            "LLhip_joint",
            "URhip_joint",
            "LRhip_joint",
        ]
        if not all(j in msg.name for j in required_joints):
            return

        UL_idx = msg.name.index("ULhip_joint")
        LL_idx = msg.name.index("LLhip_joint")
        UR_idx = msg.name.index("URhip_joint")
        LR_idx = msg.name.index("LRhip_joint")

        ULpos = (
            msg.position[UL_idx] if len(msg.position) > UL_idx else 0.0
        )  # Left Upper
        LLpos = (
            msg.position[LL_idx] if len(msg.position) > LL_idx else 0.0
        )  # Left Lower
        URpos = (
            msg.position[UR_idx] if len(msg.position) > UR_idx else 0.0
        )  # Right Upper
        LRpos = (
            msg.position[LR_idx] if len(msg.position) > LR_idx else 0.0
        )  # Right Lower

        a, b, c = 142.5, 315.0, 324.0

        # Reconstruct internal linkage angles from URDF joint positions
        tL4 = ULpos + (np.pi / 2)
        tL1 = LLpos + 1.57

        EL = 2 * b * (c + a * (np.cos(tL4) - np.cos(tL1)))
        FL = 2 * a * b * (np.sin(tL4) - np.sin(tL1))
        GL = (
            (c**2)
            + 2 * (a**2)
            + (2 * c * a * np.cos(tL4))
            - (2 * c * a * np.cos(tL1))
            - (2 * (a**2) * np.cos(tL4 - tL1))
        )

        denomL = GL - EL
        if abs(denomL) < 1e-6:
            denomL = 1e-6

        phiL = 2 * np.arctan((-FL + np.sqrt(max(0.0, EL**2 + FL**2 - GL**2))) / denomL)

        xL_calc = c + (a * np.cos(tL4)) + (b * np.cos(phiL))
        yL_calc = (a * np.sin(tL4)) + (b * np.sin(phiL))

#        self.get_logger().info(
#            f"FK Computed Position -> Left Leg (X: {xL_calc:.2f} mm, Y: {yL_calc:.2f} mm)"
#        )


def main(args=None):
    rclpy.init(args=args)
    node = IKtester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
