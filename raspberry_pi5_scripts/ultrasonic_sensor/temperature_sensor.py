import bme280
import smbus2


class TemperatureSensor:
    def __init__(self, i2c_adrress=0x77, name=None):
        # BME280 sensor address (default address)
        self.address = i2c_adrress

        self.name = name

        # Initialize I2C bus
        self.bus = smbus2.SMBus(1)

        # Load calibration parameters
        self.calibration_params = bme280.load_calibration_params(self.bus, self.address)

        # Extract temperature, pressure, humidity, and corresponding timestamp
        self.temperature_celsius = None

    def read(self):
        "This function returns the Temperature in Celcius degrees."
        data = bme280.sample(self.bus, self.address, self.calibration_params)
        return data.temperature

    def __repr__(self):
        return f"{type(self)},name: {self.name},address:{self.address}"
