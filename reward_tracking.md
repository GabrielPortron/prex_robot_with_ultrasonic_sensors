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
reward = -(distance_to_center + 0.5 *  norm(action - last_action))
bonus_reward = 100
malus_reward = 10
```

![](reward_stats/Reward_6/Sum.png)