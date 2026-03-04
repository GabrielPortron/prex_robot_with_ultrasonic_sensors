import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Odometry

from geometry_msgs.msg import Twist
import numpy as np
import time


from tf2_ros import TransformListener, Buffer
from tf2_msgs.msg import TFMessage
# from tf_transformations import euler_from_quaternion
import math


# class YawFromTFNode(Node):
#     def __init__(self, verbose=False):
#         super().__init__("yaw_from_tf_node")
#         self.verbose = verbose
#         # Create a buffer and listener to handle transforms
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # Subscribe to /tf topic to receive transformation messages
#         self.subscription = self.create_subscription(
#             TFMessage, "/tf", self.tf_callback, 10
#         )
#         self.yaw = 0.0

#     def tf_callback(self, msg):
#         # Iterate through all transforms received in the TFMessage
#         for transform in msg.transforms:
#             # Check if this transform is the one we're looking for (child_frame_id: 'LHR-2D695193')
#             if transform.child_frame_id == "LHR-2D695193":
#                 # Extract quaternion
#                 quat = transform.transform.rotation
#                 q = [quat.x, quat.y, quat.z, quat.w]

#                 # Convert quaternion to Euler angles (roll, pitch, yaw)
#                 roll, pitch, yaw = euler_from_quaternion(q)
#                 self.yaw = yaw  # math.degrees(yaw)

#                 if self.verbose:
#                     # Log yaw in radians and degrees
#                     self.get_logger().info(f"Yaw from transform: {yaw} radians")
#                     self.get_logger().info(
#                         f"Yaw in degrees: {math.degrees(yaw)} degrees"
#                     )


class MyNode(Node):

    def __init__(
        self,
        topic_sub="/prex/sensor_data",
        topic_pub="/prex/cmd_vel",
        type_ros2_msg="String",
        verbose=False,
    ):
        super().__init__("my_node")
        self.verbose = verbose
        self.state = None
        self.type_ros2_msg = type_ros2_msg
        if type_ros2_msg == "String":
            self.publisher = self.create_publisher(String, topic_pub, 1)
            self.subscription = self.create_subscription(
                String, topic_sub, self.read_state, 10
            )
        else:
            self.publisher = self.create_publisher(Twist, topic_pub, 1)
            self.subscription = self.create_subscription(
                Float32MultiArray, topic_sub, self.read_state, 10
            )

        self.subscription  # prevent unused variable warning

    def read_state(self, msg):
        if self.type_ros2_msg == "String":
            try:
                data = list(map(float, msg.data.split("#")[1:]))
                self.state = np.array(data)
                if self.verbose:
                    print(f"I recieved: {self.state}")
            except:
                if self.verbose:
                    print("state not update!")
        else:
            try:
                self.state[:4] = np.multiply(msg.data[:4], 0.01)
                if self.verbose:
                    print(f"I recieved: {self.state}")
            except:
                if self.verbose:
                    print("state not update!")

    # def send_message(self, action: str):

    def send_message(self, action):
        if isinstance(action, str) and action == "reset":
            msg = String()
            msg.data = action
            self.publisher.publish(msg)
        else:
            if self.type_ros2_msg == "String":
                try:
                    msg = String()
                    msg.data = str(action)
                    self.publisher.publish(msg)
                    if self.verbose:
                        self.get_logger().info('Publishing: "%s"' % msg.data)

                except Exception as e:
                    if self.verbose:
                        print(e)
            else:
                try:

                    linear_velocity, angular_velocity = action
                    # Create a Twist message
                    msg = Twist()

                    # Set linear and angular velocities
                    msg.linear.x = (
                        linear_velocity  # Set forward/backward linear velocity
                    )
                    msg.linear.y = 0.0
                    msg.linear.z = 0.0

                    msg.angular.x = 0.0
                    msg.angular.y = 0.0
                    msg.angular.z = angular_velocity  # Set rotational angular velocity around z-axis

                    # Publish the message
                    self.publisher.publish(msg)
                    if self.verbose:
                        self.get_logger().info(
                            'Publishing linear velocity: "%s", angular velocity: "%s"'
                            % (msg.linear.x, msg.angular.z)
                        )

                except Exception as e:
                    if self.verbose:
                        print(e)
        rclpy.spin_once(self)


class NodeRealRobot(MyNode):
    def __init__(
        self,
        topic_sub_odom="/odom",
        topic_sub="/prex/sensor_data",
        topic_pub="/prex/cmd_vel",
        type_ros2_msg="String",
        verbose=False,
    ):
        super().__init__(topic_sub, topic_pub, type_ros2_msg, verbose)
        self.subscription_odom = self.create_subscription(
            Odometry, topic_sub_odom, self.read_odometry, 10
        )
        self.state = np.zeros(10)  # 4 ultrason + linear.xyz +angular.xyz
        self.subscription_odom  # prevent unused variable warning

    def read_odometry(self, msg):

        try:
            data = np.zeros(6)
            data[0] = msg.twist.twist.linear.x
            data[1] = msg.twist.twist.linear.y
            data[2] = msg.twist.twist.linear.z
            data[3] = msg.twist.twist.angular.x
            data[4] = msg.twist.twist.angular.y
            data[5] = msg.twist.twist.angular.z

            self.state[4:] = data

            if self.verbose:
                print(f"I recieved: {self.state}")
        except:
            if self.verbose:
                print("state not update!")


class PrexWorld:
    def __init__(
        self,
        max_episode_length=1000,
        out_of_range=4,
        too_close=0.3,
        perimeter=(1.625 * 2, 0.925 * 2),
        max_linear_speed=0.7,
        max_angular_speed=0.4,
        topic_sub="/prex/sensor_data",
        topic_sub_odom=None,
        topic_pub="/prex/cmd_vel",
        type_ros2_msg="Twist",
        device="cuda",
        dt=0.005,
        verbose=True,
        max_random_steps=50,
        radius_target=0.3,
        real_robot=False,
        time_factor=10,
        clipping_limit=20,
        max_speed_bonus=5.0,
        repeating_action=1,
        initial_theta=0,
        size_robot=0.30,
    ):
        # env definition
        self.too_close = too_close
        self.last_position = self.position = None
        self.theta = initial_theta
        self.repeating_action = repeating_action
        self.max_speed_bonus = max_speed_bonus
        self.can_move = True
        self.clipping_limit = clipping_limit
        self.time_to_wait = 0.3 / time_factor
        self.real_robot = real_robot
        self.radius_target = radius_target
        self.device = device
        self.verbose = verbose
        self.max_episode_length = max_episode_length
        self.step_counter = 0
        self.action_space = [2]
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        #TODO define the state space
        self.state_space = [7]        
        self.state = np.zeros(self.state_space[0])
        self.out_of_range = out_of_range
        self.too_close = too_close
        self.state = np.zeros((self.state_space[0]))
        l1, l2 = perimeter
        self.perimeter = perimeter
        self.size_robot = size_robot
        self.goal = np.array([l1 / 2, l1 / 2, l2 / 2, l2 / 2])
        self.max_distance = max(2 * l2, 2 * l1)
        self.info = {}
        self.action = None
        self.yaw_extractor_node = None
        self.max_bounds = np.array(
            (self.max_linear_speed, self.max_angular_speed), dtype=np.float32
        )  # max_linear_velocity,max_angular_velocity
        self.dt = dt
        # ros2 definition
        self.type_ros2_msg = type_ros2_msg
        if topic_sub_odom is None:
            self.node_ros2 = MyNode(
                topic_sub, topic_pub, type_ros2_msg, verbose=verbose
            )
        else:
            self.node_ros2 = NodeRealRobot(
                topic_sub_odom, topic_sub, topic_pub, type_ros2_msg, verbose=verbose
            )
            # self.yaw_extractor_node = YawFromTFNode()

        stop_action = np.array([0.0, 0.0])
        if type_ros2_msg == "String":
            stop_action = self._action_to_text(stop_action)

        self.stop_action = stop_action
        self.dist = 0
        self.timestep = 0

        # to randomize the robot's position
        self.max_random_steps = max_random_steps
        self.random_step = 0
        self.previous_linear_velocity = np.array([0.0, 0.0, 0.0])
        self.previous_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.prevoius_dist = 0.0
        self.previous_longest_dist = 0.0
        self.previous_state = np.array([0.3, 0.3, 0.3, 0.3])
        self.rotate = False
        self.moves = np.array([False, False, False, False])

    def _action_to_text(self, action):
        if action.shape == (1, 2):
            linear_vel, angular_vel = action[0]
            return str(linear_vel) + "#" + str(angular_vel)
        else:
            linear_vel, angular_vel = action
            return str(linear_vel) + "#" + str(angular_vel)

    def step(self, action: str):
        self.step_counter += 1

        # read current robot state
        self.read_robot_state()

        # check if it is a good action
        self.controller(action.copy(), self.position)[0]
        self.action_controlled = action  # self.controller(action, self.position)[0]

        # send the action
        if self.type_ros2_msg == "String":
            self.action_controlled = (
                self._action_to_text(self.action_controlled) + "#" + str(self.timestep)
            )
        # print("action controlled:", self.action_controlled)
        self.node_ros2.send_message(self.action_controlled)
        t0 = time.time()
        while time.time() - t0 <= self.time_to_wait:
            rclpy.spin_once(self.node_ros2)

        if self.type_ros2_msg == "String":
            self.node_ros2.send_message(self.stop_action + "#" + str(self.timestep))
        else:
            self.node_ros2.send_message(self.stop_action)

        t0 = time.time()
        while time.time() - t0 <= self.time_to_wait:
            rclpy.spin_once(self.node_ros2)

        if self.verbose:
            print(action)

        # read the new state after the action has been performed
        self.read_robot_state(action)
        self.action = action
        reward, done = self._compute_reward(self.state, self.action)
        self.timestep += 1

        return self.state, reward, self.info, done

    def check_and_correct_value(self, state, min_dist=0.3):

        if state[0] != min_dist:

            diff = self.perimeter[0] - state[0]
            if diff > min_dist and state[1] == min_dist:
                state[1] = diff
        elif state[1] != min_dist:
            diff = self.perimeter[0] - state[1]
            if diff > min_dist and state[0] == min_dist:
                state[0] = diff

        if state[2] != min_dist:
            diff = self.perimeter[1] - state[2]
            if diff > min_dist and state[3] == min_dist:
                state[3] = diff
        elif state[3] != min_dist:
            diff = self.perimeter[1] - state[3]
            if diff > min_dist and state[2] == min_dist:
                state[2] = diff


    def read_robot_state(self, action=None):
        rclpy.spin_once(self.node_ros2)
        # self.node_ros2.state = [d1,d2,d3,d4,vx,vy,vz,wx,wy,wz,yaw]
        state = np.round(self.node_ros2.state[:], 2)
        #TODO define the state and all the following variables as you think is best for the task
        self.state = state[[0,1,2,3,4,9,10]]
        self.linear_speed = self.state[4]
        self.angular_speed = self.state[5]
        self.dist = np.linalg.norm(self.state[0:4]**2 - self.goal**2)
        self.position = self.state[:4] #np.array([self.dist*math.cos(self.state[6]), self.dist*math.sin(self.state[6])])
        self.theta = self.state[6]/math.pi

        return self.state

    def _compute_reward(self, state: np.array, action):
        done = False
        # TODO define the reward
        reward = - (self.dist + 4 * np.linalg.norm(action)**2)

        if self.step_counter < self.max_random_steps:
            if self.dist <= self.radius_target:
                done = True
                self.info["terminate"] = "it reached the goal"
                #TODO consider wether or not to reward the robot if it completed the task
                reward += 100
        else:
            done = True
            self.info["terminate"] = "it reached max episode length"
            #TODO consider wether or not to penalize the robot if it did not complete the task
            reward -= 10
        return (
            reward,
            done,
        )

    def reset(self):
        self.previous_state[:4] = 0.3
        self.rotate = False
        self.moves[:] = False
        self.previous_longest_dist = 0.0
        self.prevoius_dist = 0.0
        self.previous_linear_velocity[:] = 0.0
        self.previous_angular_velocity[:] = 0.0
        self.random_step = 0
        self.step_counter = 0
        self.dist = 0
        self.read_robot_state()
        self.last_position = self.position.copy()

        if self.type_ros2_msg == "String":
            self.node_ros2.send_message("reset")
            self.read_robot_state()

            while sum(self.last_position == self.position) == 4:
                self.node_ros2.send_message("reset")
                self.read_robot_state()

        self.action = None
        reward, done = self._compute_reward(self.state, np.array([0.0, 0.0]))
        self.theta = self.state[6]/math.pi
        done = False

        self.info.clear()

        return self.state, reward, self.info, done

    def controller(self, action, state, noDetectionDist=0.30):
        if self.verbose:
            print(f"action:{action}, timestep:{self.timestep}")
        linear_velocity, angular_velocity = action
        move = False
        # Initialize all movement flags as False
        st = fw = bw = rg = lf = rotate = False

        if linear_velocity > 0.0:
            fw = True
        elif linear_velocity == 0.0:
            st = True
        else:
            bw = True
        if angular_velocity > 0.0:
            lf = True
        elif angular_velocity < 0.0:
            rg = True

        if st and (rg or lf):
            rotate = True

        move_fw = move_bw = move_lf = move_rg = True

        if not rotate:
            # Example: Control the robot based on the received action
            if fw:
                if state[0] > noDetectionDist:
                    if self.verbose:
                        print("could go forward")
                    pass
                else:
                    move_fw = False
            if bw:
                if state[1] > noDetectionDist:
                    if self.verbose:
                        print("could go backward")
                        print(state[1])
                    pass
                else:
                    move_bw = False
            if lf:
                if state[2] > noDetectionDist:
                    if self.verbose:
                        print("could go left")
                        print(state[2])
                    pass
                else:
                    move_lf = False
            if rg:
                if state[3] > noDetectionDist:
                    if self.verbose:
                        print("could go right")
                        print(state[3])
                    pass
                else:
                    move_rg = False

        move = (move_fw and move_bw and move_lf and move_rg) or rotate
        if not move:

            if self.verbose:
                if fw:
                    print("FW but not moving! ")
                    print(state[0])

                if bw:
                    print("BK but not moving! ")
                    print(state[1])

                if lf:
                    print("LF but not moving! ")
                    print(state[2])

                if rg:
                    print("RG but not moving! ")
                    print(state[3])

            action[0] = 0.0
            action[1] = 0.0

        self.can_move = move
        self.moves[:] = move_fw, move_bw, move_lf, move_rg
        self.rotate = rotate
        return action, move, rotate

    def reinitialize_robot_position(self):
        print("Randominzing Robot's poistion...")
        while self.random_step < self.max_random_steps:
            action = np.random.uniform(low=0.5, high=1, size=2)
            if action[0] < 0.51:
                action[0] = 0.0
            action *= self.max_bounds * 2
            # round action
            action = np.round(action, 2)

            self.step(action)
            self.random_step += 1
        self.step_counter -= self.max_random_steps
        print("...done")
        self.read_robot_state()
