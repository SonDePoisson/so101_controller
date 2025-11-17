from mink.contrib import TeleopMocap, keycodes


class Teleop(TeleopMocap):
    """
    Teleop générique : reprend TeleopMocap
    mais change la touche de bascule rotation (KEY_PERIOD → KEY_SPACE).
    """

    def __init__(self, data):
        super().__init__(data)

        self.actions[keycodes.KEY_SPACE] = self.toggle_rotation
