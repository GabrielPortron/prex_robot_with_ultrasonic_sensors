# **Reward Tracking Document**

This document introduces the path followed to design a relevant and efficient reward function for the Prex.
The different reward functions will follow the same structure : 
- A main function
- A bonus reward in case of success
- A malus reward in case of failure

```
self.state_space = [7] 

# self.node_ros2.state = [d1,d2,d3,d4,vx,vy,vz,wx,wy,wz,yaw]
self.state = state[[0,1,2,3,4,9,10]]
self.linear_speed = self.state[4]
self.angular_speed = self.state[5]
self.position = self.state[:4]
self.dist = np.linalg.norm(self.position**2 - self.goal**2)
self.theta = self.state[6]/math.pi
```

## First Reward - The Naive One

The idea with this first test was to have a really simple function to discover the behaviour of the robot :

```
reward = -distance_to_center
bonus_reward = 1000
malus_reward = 10
```
![](reward_stats/Reward_1/Naive.png)

## Second Rewards - Adding Linear and angular velocities

Now, the robot also needs to have continuous actions. We don't want it to make brutal movements all the time. This is why the first idea to solve this problem is to have the velocities appearing in the reward function.

```
reward = -(distance_to_center + linear_speed)
bonus_reward = 1000
malus_reward = 10
```
![](reward_stats/Reward_2/Vx.png)

```
reward = -(distance_to_center + linear_speed + angular_speed)
bonus_reward = 100
malus_reward = 100
```
![](reward_stats/Reward_2/Vx_and_Wz.png)

## Third Rewards - Using the action

Rather than directly using the linear and angular speed from the state vector, the idea here is to use the squared norm of the cation vector, which contains the linear and angular speeds. It allows a better control and much better results. We will try different factors for this term :

```
reward = -(distance_to_center + norm(action)**2)
bonus_reward = 100
malus_reward = 10
```
![](reward_stats/Reward_3/One.png)

```
reward = -(distance_to_center + 2*norm(action)**2)
bonus_reward = 100
malus_reward = 10
```
![](reward_stats/Reward_3/Two.png)

```
reward = -(distance_to_center + 4*norm(action)**2)
bonus_reward = 100
malus_reward = 10
```
![](reward_stats/Reward_3/Four.png)

## Fourth Rewards - Playing with bonus

Now that we found a great upgrade to the main function, we can try to tweak the bonus reward to see its influence :

```
reward = -(norm(action)**2)
bonus_reward = 100 - distance_to_center
malus_reward = 10
```
![](reward_stats/Reward_4/Difference.png)

```
reward = -(norm(action)**2)
bonus_reward = 1 / (distance_to_center + 0.01)
malus_reward = 10
```
![](reward_stats/Reward_4/Inverse.png)

```
reward = -(distance_to_center * norm(action)**2)
bonus_reward = 100
malus_reward = 10
```
![](reward_stats/Reward_4/Product.png)

## Fifth Rewards - Using the difference of actions

The objective is the same that the one of the third part but rather than trying to minimize the actions, we focus on the difference between two consecutives actions. And the same way that we did in the last part, we will play with the bonus : 

```
reward = -(norm(action - last_action)**2)
bonus_reward = 100 - distance_to_center
malus_reward = 10
```
![](reward_stats/Reward_5/Difference.png)

```
reward = -(distance_to_center * norm(action - last_action)**2)
bonus_reward = 100
malus_reward = 10
```
![](reward_stats/Reward_5/Product.png)

## Sixth Rewards - Adding the action in the state

This part led to a lot of problems. Indeed, adding the action in the state made that from a certain amount of time, the robot stop to learn and goes in the corners without moving. The following graphs will show a decay in the behavior of the robot while trying new reward functions : 

```
self.state_space = [7+2] 

# self.node_ros2.state = [d1,d2,d3,d4,vx,vy,vz,wx,wy,wz,yaw]
self.state[:-2] = state[[0,1,2,3,4,9,10]]
self.state[-2:] = action
self.linear_speed = self.state[4]
self.angular_speed = self.state[5]
self.position = self.state[:4]
self.dist = np.linalg.norm(self.position**2 - self.goal**2)
self.theta = self.state[6]/math.pi

#red
reward = -(norm(action - last_action)**2)
bonus_reward = 100 - distance_to_center
malus_reward = 10

#green
reward = -(distance_to_center * norm(action - last_action)**2)
bonus_reward = 100
malus_reward = 10
```

![](reward_stats/Reward_6/Comparaison_6.png)

```
reward = -(distance_to_center + norm(action - last_action))
bonus_reward = 100
malus_reward = 10
```

![](reward_stats/Reward_6/Mono.png)

## Seventh Rewards - Adding the higher derivatives

Here, we want to try to add the acceleration, jerk, snap, ... of the robot to try to smoothen the movement. A new term appears in the base reward function :

The first one still had the action in the state

```
reward = 10 / (distance_to_center + 0.01) - delta_act - hderivatives

bonus_reward = 100
malus_reward = 1
```

![](reward_stats/Reward_7/Action.png)

This one doesn't

![](reward_stats/Reward_7/No_action.png)

```
reward = 10 / (distance_to_center + 0.01) - hderivatives

bonus_reward = 100
malus_reward = 1
```

![](reward_stats/Reward_7/No_delta_action.png)

Here is the comparison for this section :

![](reward_stats/Reward_7/Comparison.png)

## Eigth Rewards - More about the higher derivatives

Now that we have the idea on how the higher derivatives work, we can change it to be more precise. We know tkae the linear and angular speeds as starting points to have more precise data. Also, we compute the derivatives up to the pop (5th derivative of speed). The first ones are without the controller :

```
reward = 10 / (distance_to_center + 0.01) - delta_act - hderivatives_v - hderivatives_w

bonus_reward = 100
malus_reward = 1
```

![](reward_stats/Reward_8/Difference_1.png)
![](reward_stats/Reward_8/Difference_2.png)
![](reward_stats/Reward_8/Difference_3.png)

```
reward = 10 / (distance_to_center + 0.01) 
    + 1 / (delta_act + 0.5)
    + 1 / (hderivatives_v + 0.5)
    + 1 / (hderivatives_w + 0.5)

bonus_reward = 100
malus_reward = 1
```

![](reward_stats/Reward_8/Inverse_1.png)
![](reward_stats/Reward_8/Inverse_2.png)
![](reward_stats/Reward_8/Inverse_3.png)

Next, I tried new factors for the term in `1 / dist` : 20 (red), 40 (green) and 100 (purple):

![](reward_stats/Reward_8/Comparison_1.png)
![](reward_stats/Reward_8/Comparison_2.png)
![](reward_stats/Reward_8/Comparison_3.png)

Then I put the controller back :



