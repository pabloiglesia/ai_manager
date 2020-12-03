"""
This class defines a RL environment for a pick and place task with a UR3 robot.
This environment is defined by its center (both cartesian and angular coordinates), the total length of its x and y axis
and other parameters
"""

import random

class Environment:
    X_LENGTH = 0.195  # Total length of the x axis environment in meters
    Y_LENGTH = 0.26  # Total length of the y axis environment in meters
    CARTESIAN_CENTER = [-0.31899288568, -0.00357907370787, 0.376611799631]  # Cartesian center of the RL environment
    ANGULAR_CENTER = [2.7776150703430176, -1.5684941450702112, 1.299912452697754, -1.3755658308612269,
                      -1.5422008673297327, -0.3250663916217249]  # Angular center of the RL environment
    PLACE_CARTESIAN_CENTER = [0, 0.25, CARTESIAN_CENTER[2]]  # Cartesian center of the place box
    ANGULAR_PICTURE_PLACE = [1.615200161933899, -1.235102955495016, 0.739865779876709, -1.2438910643206995, -1.5095704237567347, -0.06187755266298467]

    PICK_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action
    ACTION_DISTANCE = 0.01  # Distance to the object when the robot is performing the pick and place action

    ENV_BOUNDS_TOLERANCE = 0.01

    @staticmethod
    def generate_random_state():
        """
        Calculates random coordinates inside the Relative Environment defined
        :return:
        """
        coordinate_x = random.uniform((-Environment.X_LENGTH + Environment.ENV_BOUNDS_TOLERANCE) / 2,
                                      (Environment.X_LENGTH - Environment.ENV_BOUNDS_TOLERANCE) / 2)
        coordinate_y = random.uniform((-Environment.Y_LENGTH + Environment.ENV_BOUNDS_TOLERANCE) / 2,
                                      (Environment.Y_LENGTH - Environment.ENV_BOUNDS_TOLERANCE) / 2)
        return [coordinate_x, coordinate_y]

    @staticmethod
    def get_relative_corner(corner):
        """
        Function used to calculate the coordinates of the environment corners relative to the CARTESIAN_CENTER.

        :param corner: it indicates the corner that we want to get the coordinates. It' s composed by two letters
        that indicate the cardinality. For example: ne indicates North-East corner
        :return coordinate_x, coordinate_y:
        """
        if corner == 'sw' or corner == 'ws':
            return -Environment.X_LENGTH / 2, Environment.Y_LENGTH / 2
        if corner == 'nw' or corner == 'wn':
            return Environment.X_LENGTH / 2, Environment.Y_LENGTH / 2
        if corner == 'ne' or corner == 'en':
            return Environment.X_LENGTH / 2, -Environment.Y_LENGTH / 2
        if corner == 'se' or corner == 'es':
            return -Environment.X_LENGTH / 2, -Environment.Y_LENGTH / 2

    @staticmethod
    def is_terminal_state(coordinates, object_gripped):
        """
        Function used to determine if the current state of the robot is terminal or not
        :return: bool
        """
        def get_limits(length): return length / 2 - Environment.ENV_BOUNDS_TOLERANCE  # functon to calculate the box boundaries
        x_limit_reached = abs(coordinates[0]) > get_limits(Environment.X_LENGTH)  # x boundary reached
        y_limit_reached = abs(coordinates[1]) > get_limits(Environment.Y_LENGTH)  # y boundary reached
        return x_limit_reached or y_limit_reached or object_gripped # If one or both or the boundaries are reached --> terminal state