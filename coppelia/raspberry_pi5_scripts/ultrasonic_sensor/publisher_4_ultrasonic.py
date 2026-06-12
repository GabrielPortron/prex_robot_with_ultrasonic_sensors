import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from gpiozero import DistanceSensor


class MyNode(Node):

    def __init__(self):
        super().__init__("my_node")
        self.publisher_ = self.create_publisher(Float32MultiArray, "chatter", 10)
        timer_period = 0.100  # seconds
        self.get_logger().info("Starting:")
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.sensor1 = DistanceSensor(echo=27, trigger=17, queue_len=1, partial=True)
        self.sensor2 = DistanceSensor(echo=24, trigger=25, queue_len=1, partial=True)
        self.sensor3 = DistanceSensor(echo=13, trigger=26, queue_len=1, partial=True)
        self.sensor4 = DistanceSensor(echo=22, trigger=23, queue_len=1, partial=True)
        self.msg = Float32MultiArray()
        self.msg.data = [0.0, 0.0, 0.0, 0.0]

    def timer_callback(self):
        distance1 = 0.0
        distance2 = 0.0
        distance3 = 0.0
        distance4 = 0.0

        try:
            distance1 = (
                self.sensor1.distance * 100
            )  # Measure distance and convert from meters to centimeters
        except:
            self.get_logger().warn("No measure sensor1")
            print("No measure sensor1")

        try:
            distance2 = (
                self.sensor2.distance * 100
            )  # Measure distance and convert from meters to centimeters
        except:
            self.get_logger().warn("No measure sensor2")
            print("No measure sensor2")

        try:
            distance3 = (
                self.sensor3.distance * 100
            )  # Measure distance and convert from meters to centimeters
        except:
            self.get_logger().warn("No measure sensor3")
            print("No measure sensor3")

        try:
            distance4 = (
                self.sensor4.distance * 100
            )  # Measure distance and convert from meters to centimeters
        except:
            self.get_logger().warn("No measure sensor4")
            print("No measure sensor4")

        self.msg.data[0] = distance1
        self.msg.data[1] = distance2
        self.msg.data[2] = distance3
        self.msg.data[3] = distance4

        self.publisher_.publish(self.msg)
        self.get_logger().info(
            f"Publishing: {self.msg.data[0]},{self.msg.data[1]},{self.msg.data[2]},{self.msg.data[3]}"
        )


def main(args=None):

    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    print("starting")
    main()
