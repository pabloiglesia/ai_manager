
"""
This class defines a RL environment for a pick and place task with a UR3 robot.
This environment is defined by its center (both cartesian and angular coordinates), the total length of its x and y axis
and other parameters
"""


class Environment:
    def __init__(self):
        self.x_length = 0.30  # Total length of the x axis environment in meters
        self.y_length = 0.44 # Total length of the y axis environment in meters
        self.cartesian_center = [-0.31899288568, -0.00357907370787, 0.226626573286]
        self.angular_center = [2.7776150703430176, -1.5684941450702112, 1.299912452697754, -1.3755658308612269,
                              -1.5422008673297327, -0.3250663916217249]
        self.pick_distance = 0.01  # Distance to the object when the robot is performing the pick and place action