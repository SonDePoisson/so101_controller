from so101_driver.so101_driver import SO101Driver
import os

if __name__ == "__main__":
    device = os.getenv("SO101_PORT", "/dev/ttyUSB0")
    driver = SO101Driver(device)
    driver.free_robot()
    driver.calibrate_joints()
    driver.free_robot()