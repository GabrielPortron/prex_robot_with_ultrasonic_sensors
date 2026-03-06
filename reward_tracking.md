# **Reward Tracking Document**

This document introduces the path followed to design a relevant and efficient reward function for the Prex.
The different reward functions will follow the same structure : 
- A main function
- A bonus reward in case of success
- A malus reward in case of failure

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

