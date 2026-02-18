sim = require 'sim'
simROS2 = require 'simROS2'

local publisher
local subscriber
local robot = sim.getObject('..')
local startTime_sensors = sim.getSimulationTime()
local sleepDuration_sensors = 0.1 -- 10Hz
local startTime_motors = sim.getSimulationTime()
local sleepDuration_motors = 0.1 -- 10Hz

-- Define a function to reset robot position randomly
function resetRobotPosition()
    -- Define the range for random position (adjust as needed)
    local minX, maxX = -1, 0.45
    local minY, maxY = -0.7, 0.7
    local minZ, maxZ = 0, 0

    -- Generate random position within the range
    local randomX = math.random() * (maxX - minX) + minX
    local randomY = math.random() * (maxY - minY) + minY
    local randomZ = math.random() * (maxZ - minZ) + minZ

    -- Set the random position
    sim.setObjectPosition(robot, -1, {randomX, randomY, 0.13879})

    local minYaw, maxYaw = 0, 360
    -- Optionally, set a random orientation (yaw, pitch, roll)
    local randomRoll = 0
    local randomPitch = 0
    local randomYaw = math.random() * (maxYaw - minYaw) + minYaw

    -- Set the random orientation
    sim.setObjectOrientation(robot, -1, {randomRoll, randomPitch, 0.0})
    
    -- Reset the robot's velocity and acceleration to zero
    sim.setJointTargetVelocity(motorLeft, 0.0)   -- Convert to motor speed (rad/s)
    sim.setJointTargetVelocity(motorRight, 0.0)
end

function sysCall_init()
    -- Initialize ROS 2 Publisher (advertise the topic with std_msgs/String)
    publisher = simROS2.createPublisher('/prex/sensor_data', 'std_msgs/msg/String')

    -- Initialize ROS 2 subscriber for velocity commands
    subscriber = simROS2.createSubscription("/prex/action", 'std_msgs/msg/String',"actionCallback")

    -- Initialize robot components
    local obstacles = sim.createCollection(0)
    sim.addItemToCollection(obstacles, sim.handle_all, -1, 0)
    sim.addItemToCollection(obstacles, sim.handle_tree, robot, 1)

    usensors = {}
    usensors[1] = sim.getObject("/" .. "ultrasonicSensor_front")
    usensors[2] = sim.getObject("/" .. "ultrasonicSensor_back")
    usensors[3] = sim.getObject("/" .. "ultrasonicSensor_left")
    usensors[4] = sim.getObject("/" .. "ultrasonicSensor_right")

    motorLeft = sim.getObject("../leftMotor")
    motorRight = sim.getObject("../rightMotor")
    noDetectionDist = 0.30
    maxDetectionDist = 4
    detect = { 0, 0, 0, 0 }
    linear_velocity = 0.0
    angular_velocity = 0.0
    timestep = 0.0
    dt = 0.050
end

function sysCall_cleanup()
    -- Cleanup code (optional)
    simROS2.shutdownPublisher(publisher)
    simROS2.shutdownSubscription(subscriber)
end

function sysCall_actuation()
    -- Read the sensor values and update the 'detect' array
    for i = 1, 4 do
        local res, dist = sim.readProximitySensor(usensors[i])
        if (res > 0) then
            if (dist < maxDetectionDist) then
                detect[i] = dist
            else
                -- out of range
                detect[i] = maxDetectionDist
            end
        else
            -- too close
            detect[i] = noDetectionDist
        end
    end
    
    -- Get the velocity (linear velocity: x, y, z, angular velocity: rx, ry, rz)
    local linearVelocity, angularVelocity = sim.getObjectVelocity(robot)

    -- Create a string from the 'detect' array
    local detectStr = "Sensor readings: "
    local msg = ""
    for i = 1, #detect do
        detectStr = detectStr .. "Sensor " .. i .. ": " .. detect[i] .. "  "
        msg = msg .. "#" .. detect[i] .. " "
    end
    for i = 1, #linearVelocity do
        msg = msg .. "#" .. linearVelocity[i] .. " "
    end
    for i = 1, #angularVelocity do
        msg = msg .. "#" .. angularVelocity[i] .. " "
    end

    if   sim.getSimulationTime() - startTime_sensors >= sleepDuration_sensors then
        -- Publish the sensor data
        simROS2.publish(publisher, { data = msg })
        startTime_sensors = sim.getSimulationTime()
    end
end

-- Callback function for the /action subscriber
function actionCallback(msg)
    if   sim.getSimulationTime() - startTime_motors >= sleepDuration_motors then
        -- Process the incoming message from the /action topic
        local action = msg.data
        print("Received action: " .. action)
        if action == "reset" then
                resetRobotPosition()

        else
            if action then
                local linear_velocity, angular_velocity, timestep = action:match("([+-]?%d*%.%d+)#([+-]?%d*%.%d+)#([+-]?%d+)")
                -- Convert them to numbers
                linear_velocity = tonumber(linear_velocity)
                angular_velocity = tonumber(angular_velocity)
                dt = tonumber(dt)
            
                -- Define robot parameters (adjust these values to match your robot's configuration)
                local wheel_radius = 0.05  -- Adjust based on your robot's actual wheel radius (meters)
                local wheel_distance = 0.2 -- Adjust based on the distance between the wheels (meters)

                -- Compute the left and right wheel velocities (differential drive kinematics)
                local v_left = linear_velocity - angular_velocity * wheel_distance / (2*wheel_radius)
                local v_right = linear_velocity + angular_velocity * wheel_distance / (2*wheel_radius)

                -- Example: Control the robot based on the received action
                sim.setJointTargetVelocity(motorLeft, v_left / wheel_radius)   -- Convert to motor speed (rad/s)
                sim.setJointTargetVelocity(motorRight, v_right / wheel_radius)   -- Convert to motor speed (rad/s)

                -- You can call the reset function here when needed, for example:

            end
        end
        startTime_motors = sim.getSimulationTime()
    end
end
