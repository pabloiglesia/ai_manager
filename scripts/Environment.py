"""
This class defines a RL environment for a pick and place task with a UR3 robot.
This environment is defined by its center (both cartesian and angular coordinates), the total length of its x and y axis
and other parameters
"""


class Environment:
    X_LENGTH = 0.30  # Total length of the x axis environment in meters
    Y_LENGTH = 0.44  # Total length of the y axis environment in meters
    CARTESIAN_CENTER = [-0.31899288568, -0.00357907370787, 0.226626573286]  # Cartesian center of the RL environment
    ANGULAR_CENTER = [2.7776150703430176, -1.5684941450702112, 1.299912452697754, -1.3755658308612269,
                      -1.5422008673297327, -0.3250663916217249]  # Angular center of the RL environment
    PLACE_CARTESIAN_CENTER = [0, 0.25, 0.226626573286]  # Cartesian center of the place box

    PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action

    @staticmethod
    def is_terminal_state(coordinates):
        """
        Function used to determine if the current state of the robot is terminal or not
        :return: bool
        """
        def get_limits(length): return length / 2 - 0.01  # functon to calculate the box boundaries
        x_limit_reached = abs(coordinates[0]) > get_limits(Environment.X_LENGTH)  # x boundary reached
        y_limit_reached = abs(coordinates[1]) > get_limits(Environment.Y_LENGTH)  # y boundary reached
        return x_limit_reached or y_limit_reached  # If one or both or the boundaries are reached --> terminal state

    @staticmethod
    def get_relative_corner(corner):
        if corner == 0:
            return -Environment.X_LENGTH / 2, Environment.Y_LENGTH / 2
        if corner == 1:
            return Environment.X_LENGTH / 2, Environment.Y_LENGTH / 2
        if corner == 2:
            return Environment.X_LENGTH / 2, -Environment.Y_LENGTH / 2
        if corner == 3:
            return -Environment.X_LENGTH / 2, -Environment.Y_LENGTH / 2