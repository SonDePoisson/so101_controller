from st3215 import ST3215
import yaml
import time
import os
import sys
import select


class SO101Driver:
    """
    Driver class for controlling the SO101 robot using the st3215 library.
    """

    def __init__(self, device):
        """
        Initialize the connection to the SO101 robot serial bus.

        :param device: Serial port path (e.g., '/dev/ttyUSB0')
        """
        try:
            self.st = ST3215(device)
            self.servo_ids = self.st.ListServos()
        except Exception as e:
            print("SO101Driver init:", e)
            raise

    def free_robot(self):
        for id in self.servo_ids:
            self.st.StopServo(id)

    def activate_robot(self):
        for id in self.servo_ids:
            self.st.StartServo(id)

    def move_servo(self, servo_id, position, speed=2400, acc=50, wait=False):
        """
        Move a servo to a target position.

        :param servo_id: ID of the servo to move
        :param position: Target position
        :param speed: Movement speed
        :param acc: Acceleration
        :param wait: Wait for movement to finish
        """
        return self.st.MoveTo(servo_id, position, speed, acc, wait)

    def read_servo_position(self, servo_id):
        """
        Read the current position of a servo.

        :param servo_id: ID of the servo to read
        :return: Position value or None if error
        """
        try:
            position = self.st.ReadPosition(servo_id)
            return position
        except Exception as e:
            print(f"Error reading position for servo {servo_id}:", e)
            return None

    def dump_servo_state(self, servo_id):
        """
        Read and display the state of a servo: status, position, velocity, acceleration.

        :param servo_id: ID of the servo to read
        :return: dict with status, position, velocity, acceleration
        """
        try:
            position = self.st.ReadPosition(servo_id)
            velocity = self.st.ReadSpeed(servo_id)
            acceleration = self.st.ReadAccelaration(servo_id)
            status = self.st.ReadStatus(servo_id)
        except Exception as e:
            print(f"Error reading servo {servo_id} state:", e)
            return None

        state = {
            "id": servo_id,
            "status": status,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
        }

        # Présentation propre dans le terminal
        print(f"\nServo {servo_id} state:")
        print(f"  Status      : {status}")
        print(f"  Position    : {position}")
        print(f"  Velocity    : {velocity}")
        print(f"  Acceleration: {acceleration}")

        return state

    def calibrate_joints(self, yaml_path=None):
        """
        Let the user manually move all joints. First, all joints are set to zero and the user
        confirms the zero position. Then, the user moves the joints by hand and the method records
        the min and max position reached by each joint. At the end, it saves the results in a YAML file.

        :param yaml_path: Path to the output YAML file (default: ./joint_limits.yaml at project root)
        """
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "joint_limits.yaml")
            yaml_path = os.path.abspath(yaml_path)

        zero_positions = {}
        print("Move all joints to zero position, then press Enter")
        try:
            while True:
                for servo_id in self.servo_ids:
                    zero_positions[servo_id] = self.read_servo_position(servo_id)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    input()
                    break
        except KeyboardInterrupt:
            print("\nCalibration interrupted by user")
            return

        print("\nMove each joint by hand to its limits")
        print("Press Enter to save calibration")
        joint_limits = {servo_id: {"min": None, "max": None} for servo_id in self.servo_ids}

        try:
            while True:
                for servo_id in self.servo_ids:
                    pos = self.read_servo_position(servo_id)
                    if pos is not None:
                        if joint_limits[servo_id]["min"] is None or pos < joint_limits[servo_id]["min"]:
                            joint_limits[servo_id]["min"] = pos
                        if joint_limits[servo_id]["max"] is None or pos > joint_limits[servo_id]["max"]:
                            joint_limits[servo_id]["max"] = pos

                line = ""
                for servo_id in self.servo_ids:
                    line += f"{servo_id}: min={joint_limits[servo_id]['min']} max={joint_limits[servo_id]['max']} | "
                print(f"\r{line}", end="", flush=True)
                time.sleep(0.1)

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    input()
                    break

        except KeyboardInterrupt:
            print("\nCalibration interrupted by user")
            return

        # Stocke tout dans le YAML
        calibration_data = {"zero": zero_positions, "limits": joint_limits}
        with open(yaml_path, "w") as f:
            yaml.dump(calibration_data, f)
        print(f"\nJoint zeros and limits saved to {yaml_path}")
