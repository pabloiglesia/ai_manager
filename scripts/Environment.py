"""
This class defines a RL environment for a pick and place task with a UR3 robot.
This environment is defined by its center (both cartesian and angular coordinates), the total length of its x and y axis
and other parameters
"""


class Environment:
    X_LENGTH = 0.30  # Total length of the x axis environment in meters
    Y_LENGTH = 0.44  # Total length of the y axis environment in meters
    CARTESIAN_CENTER = [-0.31899288568, -0.00357907370787, 0.226626573286]
    ANGULAR_CENTER = [2.7776150703430176, -1.5684941450702112, 1.299912452697754, -1.3755658308612269,
                      -1.5422008673297327, -0.3250663916217249]
    PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action

    @staticmethod
    def is_terminal_state(self, coordinates):
        def get_limits(length): length / 2 - 0.01
        x_limit_reached = abs(coordinates[0]) > get_limits(Environment.X_LENGTH)
        y_limit_reached = abs(coordinates[1]) > get_limits(Environment.Y_LENGTH)
        return x_limit_reached or y_limit_reached
