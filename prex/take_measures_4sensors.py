import argparse
import csv
import time
import lgpio as GPIO
import time


def get_distance():
    # Set TRIG LOW
    GPIO.gpio_write(h, TRIG, 0)
    time.sleep(0.03)

    # Send 10us pulse to TRIG
    GPIO.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    GPIO.gpio_write(h, TRIG, 0)
    good_measure = True
    pulse_start = pulse_end = 0.0
    # Start recording the time when the wave is sent
    while GPIO.gpio_read(h, ECHO) == 0:
        pulse_start = time.perf_counter()

    # Record time of arrival
    while GPIO.gpio_read(h, ECHO) == 1:
        pulse_end = time.perf_counter()
        if pulse_end - pulse_start >= 0.02332361516:
            distance = 400.0
            good_measure = False
            break

    if good_measure:
        # Calculate the difference in times
        pulse_duration = pulse_end - pulse_start

        # Multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = min(pulse_duration * 17150.0, 400.0)
        distance = round(distance, 4)

    return distance


def create_list_distances(n: int):
    l = []
    for _ in range(n):
        l.append(get_distance())
        # time.sleep(0.035)
    return l


class Sensor:
    import lgpio as GPIO

    def __init__(self, trig, echo, max_distance, temperature=20):
        self.trig = trig
        self.echo = echo
        self.max_distance = max_distance
        self.temperature = temperature
        self.sound_speed = 340
        # Open the GPIO chip and set the GPIO direction
        h = GPIO.gpiochip_open(0)
        GPIO.gpio_claim_output(h, self.trig)
        GPIO.gpio_claim_input(h, self.echo)
        self.update_temperature(temperature)

    def update_temperature(self, temperature):
        self.temperature = temperature
        self.sound_speed = 331 + 0.6 * self.temperature
        self.max_time = (self.max_distance / self.sound_speed) * 2

    def get_distance(self):
        # Set TRIG LOW
        GPIO.gpio_write(h, self.trig, 0)
        time.sleep(0.03)

        # Send 10us pulse to self.trig
        GPIO.gpio_write(h, self.trig, 1)
        time.sleep(0.00001)
        GPIO.gpio_write(h, self.trig, 0)
        good_measure = True
        pulse_start = pulse_end = 0.0
        # Start recording the time when the wave is sent
        while GPIO.gpio_read(h, self.echo) == 0:
            pulse_start = time.perf_counter()

        # Record time of arrival
        while GPIO.gpio_read(h, self.echo) == 1:
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
                pulse_duration * self.sound_speed * 100 / 2, self.max_distance
            )
            distance = round(distance, 4)

        return distance


if __name__ == "__main__":
    # Set pins
    TRIG = 17  # Associate pin 23 to TRIG
    ECHO = 27  # Associate pin 24 to ECHO

    # Open the GPIO chip and set the GPIO direction
    h = GPIO.gpiochip_open(0)
    GPIO.gpio_claim_output(h, TRIG)
    GPIO.gpio_claim_input(h, ECHO)
    # Determin how many meausures to take per each case
    n = input("How many distances do you want to take per each test case? ")

    # Create the list of angle to test
    angles = input(
        "Type all the angles you want to test separate by a comma. i.e. 10,20: "
    ).split(",")
    print(f"Here is the list of angles: {angles}")

    # Create the list of distancese to test
    distances = input(
        "Type all the distances (cm) you want to test separate by a comma. i.e. 10,20: "
    ).split(",")
    print(f"Here is the list of distances: {distances}")

    # Decide weather iterating distances over angles or the opposit
    angle_first = input(
        "Type 0 to test more distances for the same angle, 1 for the opposit: "
    )

    # Choose a filename for the csv file
    name_file = input("Write the name for your csv file without the extension .csv: ")

    with open(name_file + ".csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["angle_distance", "measure"])

    # Fill the csv file
    if angle_first == "0":
        for a in angles:
            for d in distances:
                input(
                    f"Press enter to start collecting the list of {n} meausures at distance {d} cm and angle {a}"
                )
                key = a + "_" + d
                with open(name_file + ".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    for dist in create_list_distances(int(n)):
                        writer.writerow(["{angle:" + a + ",dist:" + d + "}", dist])
    else:
        for d in distances:
            for a in angles:
                input(
                    f"Press enter to start collecting the list of {n} meausures at distance {d} cm and angle {a}"
                )
                key = a + "_" + d
                with open(name_file + ".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    for dist in create_list_distances(int(n)):
                        writer.writerow(["{angle:" + a + ",dist:" + d + "}", dist])
