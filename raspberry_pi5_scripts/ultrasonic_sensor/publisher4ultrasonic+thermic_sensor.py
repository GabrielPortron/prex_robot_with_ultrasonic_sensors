import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from temperature_sensor import TemperatureSensor
import time


import lgpio as GPIO


class DistanceSensor:

    def __init__(self, trig, echo, max_distance, temperature_celsius=20):
        self.trig = trig
        self.echo = echo
        self.max_distance = max_distance
        self.temperature_celsius = temperature_celsius
        self.sound_speed = 340
        # Open the GPIO chip and set the GPIO direction
        self.h = GPIO.gpiochip_open(0)
        GPIO.gpio_claim_output(self.h, self.trig)
        GPIO.gpio_claim_input(self.h, self.echo)
        self.update_temperature_celsius(temperature_celsius)

    def update_temperature_celsius(self, temperature_celsius):
        self.temperature_celsius = temperature_celsius
        self.sound_speed = 331 + 0.6 * self.temperature_celsius
        self.max_time = 1  # (self.max_distance / self.sound_speed) * 2

    def get_distance(self):
        distance = -1
        # Set TRIG LOW
        GPIO.gpio_write(self.h, self.trig, 0)
        time.sleep(0.03)

        # Send 10us pulse to self.trig
        GPIO.gpio_write(self.h, self.trig, 1)
        time.sleep(0.00001)
        GPIO.gpio_write(self.h, self.trig, 0)
        good_measure = True
        pulse_start = pulse_end = 0.0
        # Start recording the time when the wave is sent
        while GPIO.gpio_read(self.h, self.echo) == 0:
            pulse_start = time.perf_counter()

        # Record time of arrival
        while GPIO.gpio_read(self.h, self.echo) == 1:
            pulse_end = time.perf_counter()
            if pulse_end - pulse_start >= self.max_time:
                distance = self.max_distance
                good_measure = False
                break

        if good_measure:
            # Calculate the difference in times
            pulse_duration = pulse_end - pulse_start

            # Multiply with the sonic speed (34300 cm/s)
            # and divide by 2, because there and back
            distance = min(
                pulse_duration * self.sound_speed * 100 / 2, self.max_distance * 100
            )
            distance = round(distance, 4)

        return distance


class MyNode(Node):

    def __init__(self):
        super().__init__("my_node")
        self.publisher_ = self.create_publisher(Float32MultiArray, "ultrasonics", 10)
        timer_period = 0.100  # seconds
        self.get_logger().info("Starting:")
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.temp_sensors = [
            TemperatureSensor(0x77, "on_the_case"),
            TemperatureSensor(0x76, "on_the_stick"),
        ]
        max_distance = 2.0
        temperature_celsius = self.temp_sensors[1].read()

        self.sensor1 = DistanceSensor(
            trig=17,
            echo=27,
            max_distance=max_distance,
            temperature_celsius=temperature_celsius,
        )
        self.sensor2 = DistanceSensor(
            trig=25,
            echo=24,
            max_distance=max_distance,
            temperature_celsius=temperature_celsius,
        )
        self.sensor3 = DistanceSensor(
            trig=26,
            echo=13,
            max_distance=max_distance,
            temperature_celsius=temperature_celsius,
        )
        self.sensor4 = DistanceSensor(
            trig=23,
            echo=22,
            max_distance=max_distance,
            temperature_celsius=temperature_celsius,
        )
        self.msg = Float32MultiArray()
        self.msg.data = [0.0, 0.0, 0.0, 0.0]
        self.time_start = time.time() - 10

    def timer_callback(self):
        distance1 = 0.0
        distance2 = 0.0
        distance3 = 0.0
        distance4 = 0.0

        if time.time() - self.time_start >= 10:
            temperature_celsius1 = self.temp_sensors[0].read()
            temperature_celsius2 = self.temp_sensors[1].read()

            speed_of_sound1 = 331.3 + 0.6 * temperature_celsius1  # 343.26 # m/s
            speed_of_sound2 = 331.3 + 0.6 * temperature_celsius2  # 343.26 # m/s

            self.sensor1.speed_of_sound = speed_of_sound2
            self.sensor2.speed_of_sound = speed_of_sound2
            self.sensor3.speed_of_sound = speed_of_sound2
            self.sensor4.speed_of_sound = speed_of_sound2
            self.time_start = time.time()
            print(
                f"speed:{speed_of_sound1}, temp:{temperature_celsius1},speed:{speed_of_sound2}, temp:{temperature_celsius2}"
            )

        distance1 = self.sensor1.get_distance()
        distance2 = self.sensor2.get_distance()
        distance3 = self.sensor3.get_distance()
        distance4 = self.sensor4.get_distance()

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
