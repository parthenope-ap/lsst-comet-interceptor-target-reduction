import math

import numpy as np

from .comet_bean import CometBean
from .logger import Logger

logger = Logger()


class Comet:
    """
    A class used to represent a Comet with various orbital and positional parameters
    and methods for computing the nucleus coordinates and rotation matrix.
    """

    def __init__(self, comet_bean: CometBean) -> None:
        """
        Initializes the comet object with the given CometBean object.

        Parameters
        ----------
        comet_bean : CometBean
            The CometBean object containing the comet parameters.
        """

        # Constants
        self.__KERNEL_ROTATION_MATRIX = (3, 3)

        # Assign values directly from the CometBean object
        self.__kepler_constant_1 = comet_bean.kepler_constant_1
        self.__kepler_constant_2 = comet_bean.kepler_constant_2
        self.__obliquity_of_the_ecliptic = comet_bean.obliquity_of_the_ecliptic
        self.__anomaly = comet_bean.anomaly
        self.__ascending_node = comet_bean.ascending_node
        self.__argument_of_perihelion = comet_bean.argument_of_perihelion
        self.__inclination = comet_bean.inclination
        self.__perihelion_distance = comet_bean.perihelion_distance
        self.__eccentricity = comet_bean.eccentricity
        self.__right_ascension = comet_bean.right_ascension
        self.__declination = comet_bean.declination
        self.__true_anomaly = comet_bean.true_anomaly
        self.__geocentric_distance = comet_bean.geocentric_distance
        self.__nucleus_position = comet_bean.nucleus_position
        self.__earth_sun_distance = comet_bean.earth_sun_distance
        self.__x1 = comet_bean.x1
        self.__x2 = comet_bean.x2
        self.__y1 = comet_bean.y1
        self.__y2 = comet_bean.y2
        self.__mod_ie = comet_bean.mod_ie

        # Initialize matrices
        self.__orbital_to_image_rotation_matrix = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.__ecliptic_to_equatorial_rotation_matrix = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.__temp_matrix = np.zeros(self.__KERNEL_ROTATION_MATRIX)

        # Compute image coordinates of the nucleus
        self.__compute_nucleus_coordinates()

        # Compute the rotation matrix
        self.__compute_rotation_matrix()
        logger.info("Comet object created.")

    def __compute_nucleus_coordinates(self) -> None:
        """
        Compute the nucleus coordinates of the comet object.
        """

        logger.info("Computing nucleus coordinates...")
        u = (
            self.__earth_sun_distance
            * self.__geocentric_distance
            * math.tan(np.deg2rad(self.__nucleus_position) / np.double(3600))
        )
        self.__x1 = np.double(self.__x1) * u
        self.__y1 = np.double(self.__y1) * u
        self.__x2 = np.double(self.__x2) * u
        self.__y2 = np.double(self.__y2) * u

    def intermediate_computations(self) -> tuple[int, float]:
        """
        Perform intermediate computations for the comet object.

        Returns
        -------
        tuple[int, float]
            A tuple containing the intermediate computations.
        """
        ie = int(self.__mod_ie * self.__true_anomaly)
        ac = math.tan(0.5 * np.deg2rad(self.__true_anomaly))
        if ac < 0.0:
            ie -= 1
        return ie, ac

    def __compute_rotation_matrix(self) -> None:
        """
        Compute the rotation matrix of the comet object.
        """

        # Compute the orbital to image plane rotation matrix
        logger.info("Computing rotation matrix...")
        self.__orbital_to_image_rotation_matrix[0, 0] = math.cos(self.__ascending_node) * math.cos(
            self.__argument_of_perihelion
        ) - math.cos(self.__inclination) * math.sin(self.__argument_of_perihelion) * math.sin(
            self.__ascending_node
        )

        self.__orbital_to_image_rotation_matrix[0, 1] = -math.sin(self.__ascending_node) * math.cos(
            self.__argument_of_perihelion
        ) - math.cos(self.__inclination) * math.sin(self.__argument_of_perihelion) * math.cos(
            self.__ascending_node
        )

        self.__orbital_to_image_rotation_matrix[0, 2] = math.sin(self.__inclination) * math.sin(
            self.__argument_of_perihelion
        )

        self.__orbital_to_image_rotation_matrix[1, 0] = math.cos(self.__ascending_node) * math.sin(
            self.__argument_of_perihelion
        ) + math.cos(self.__inclination) * math.cos(self.__argument_of_perihelion) * math.sin(
            self.__ascending_node
        )

        self.__orbital_to_image_rotation_matrix[1, 1] = -math.sin(self.__ascending_node) * math.sin(
            self.__argument_of_perihelion
        ) + math.cos(self.__inclination) * math.cos(self.__argument_of_perihelion) * math.cos(
            self.__ascending_node
        )

        self.__orbital_to_image_rotation_matrix[1, 2] = -math.sin(self.__inclination) * math.cos(
            self.__argument_of_perihelion
        )

        self.__orbital_to_image_rotation_matrix[2, 0] = math.sin(self.__inclination) * math.sin(
            self.__ascending_node
        )

        self.__orbital_to_image_rotation_matrix[2, 1] = math.sin(self.__inclination) * math.cos(
            self.__ascending_node
        )

        self.__orbital_to_image_rotation_matrix[2, 2] = math.cos(self.__inclination)

        self.__temp_matrix[0, 0] = np.double(1.0)
        self.__temp_matrix[0, 1] = np.double(0.0)
        self.__temp_matrix[0, 2] = np.double(0.0)
        self.__temp_matrix[1, 0] = np.double(0.0)
        self.__temp_matrix[2, 0] = np.double(0.0)
        self.__temp_matrix[2, 1] = math.sin(self.__obliquity_of_the_ecliptic)
        self.__temp_matrix[1, 2] = -self.__temp_matrix[2, 1]
        self.__temp_matrix[1, 1] = math.cos(self.__obliquity_of_the_ecliptic)
        self.__temp_matrix[2, 2] = self.__temp_matrix[1, 1]

        for i in range(self.__KERNEL_ROTATION_MATRIX[0]):
            for k in range(self.__KERNEL_ROTATION_MATRIX[1]):
                self.__ecliptic_to_equatorial_rotation_matrix[i, k] = np.double(0.0)
                for j in range(self.__KERNEL_ROTATION_MATRIX[0]):
                    self.__ecliptic_to_equatorial_rotation_matrix[i, k] += (
                        self.__temp_matrix[i, j] * self.__orbital_to_image_rotation_matrix[j, k]
                    )

        self.__temp_matrix[0, 0] = -math.cos(self.__declination) * math.cos(self.__right_ascension)
        self.__temp_matrix[0, 1] = -math.cos(self.__declination) * math.sin(self.__right_ascension)
        self.__temp_matrix[0, 2] = -math.sin(self.__declination)
        self.__temp_matrix[1, 0] = math.sin(self.__right_ascension)
        self.__temp_matrix[1, 1] = -math.cos(self.__right_ascension)
        self.__temp_matrix[1, 2] = np.double(0.0)
        self.__temp_matrix[2, 0] = -math.sin(self.__declination) * math.cos(self.__right_ascension)
        self.__temp_matrix[2, 1] = -math.sin(self.__declination) * math.sin(self.__right_ascension)
        self.__temp_matrix[2, 2] = math.cos(self.__declination)

        for i in range(self.__KERNEL_ROTATION_MATRIX[0]):
            for k in range(self.__KERNEL_ROTATION_MATRIX[1]):
                self.__orbital_to_image_rotation_matrix[i, k] = np.double(0.0)
                for j in range(self.__KERNEL_ROTATION_MATRIX[0]):
                    self.__orbital_to_image_rotation_matrix[i, k] += (
                        self.__temp_matrix[i, j] * self.__ecliptic_to_equatorial_rotation_matrix[j, k]
                    )

    # Getter and setter methods for the attributes
    def get_header(self):
        """
        Get the header of the comet object.
        """
        return self.__header

    def set_header(self, header):
        """
        Set the header of the comet object.
        """
        self.__header = header

    def get_time_array(self):
        """
        Get the time array of the comet object.
        """
        return self.__time_array

    def set_time_array(self, time_array):
        """
        Set the time array of the comet object.
        """
        self.__time_array = time_array

    def get_orbital_to_image_rotation_matrix(self):
        """
        Get the orbital to image rotation matrix of the comet object.
        """
        return self.__orbital_to_image_rotation_matrix

    def set_orbital_to_image_rotation_matrix(self, matrix):
        """
        Set the orbital to image rotation matrix of the comet object.
        """
        self.__orbital_to_image_rotation_matrix = matrix

    def get_ecliptic_to_equatorial_rotation_matrix(self):
        """
        Get the ecliptic to equatorial rotation matrix of the comet object.
        """
        return self.__ecliptic_to_equatorial_rotation_matrix

    def set_ecliptic_to_equatorial_rotation_matrix(self, matrix):
        """
        Set the ecliptic to equatorial rotation matrix of the comet object.
        """
        self.__ecliptic_to_equatorial_rotation_matrix = matrix

    def get_temp_matrix(self):
        """
        Get the temporary matrix of the comet object.
        """
        return self.__temp_matrix

    def set_temp_matrix(self, matrix):
        """
        Set the temporary matrix of the comet object.
        """
        self.__temp_matrix = matrix

    def get_base_image(self):
        """
        Get the base image of the comet object.
        """
        return self.__base_image

    def set_base_image(self, image):
        """
        Set the base image of the comet object.
        """
        self.__base_image = image

    def get_model_image(self):
        """
        Get the model image of the comet object.
        """
        return self.__model_image

    def set_model_image(self, image):
        """
        Set the model image of the comet object.
        """
        self.__model_image = image

    def get_extended_image(self):
        """
        Get the extended image of the comet object.
        """
        return self.__extended_image

    def set_extended_image(self, image):
        """
        Set the extended image of the comet object.
        """
        self.__extended_image = image

    def get_comet_image(self):
        """
        Get the comet image of the comet object.
        """
        return self.__comet_image

    def set_comet_image(self, image):
        """
        Set the comet image of the comet object.
        """
        self.__comet_image = image

    def get_kepler_constant_1(self):
        """
        Get the first Kepler constant of the comet object.
        """
        return self.__kepler_constant_1

    def get_kepler_constant_2(self):
        """
        Get the second Kepler constant of the comet object.
        """
        return self.__kepler_constant_2

    def get_obliquity_of_the_ecliptic(self):
        """
        Get the obliquity of the ecliptic of the comet object.
        """
        return self.__obliquity_of_the_ecliptic

    def get_anomaly(self):
        """
        Get the anomaly of the comet object.
        """
        return self.__anomaly

    def set_anomaly(self, anomaly):
        """
        Set the anomaly of the comet object.
        """
        self.__anomaly = anomaly

    def get_ascending_node(self):
        """
        Get the ascending node of the comet object.
        """
        return self.__ascending_node

    def set_ascending_node(self, ascending_node):
        """
        Set the ascending node of the comet object.
        """
        self.__ascending_node = ascending_node

    def get_argument_of_perihelion(self):
        """
        Get the argument of perihelion of the comet object.
        """
        return self.__argument_of_perihelion

    def set_argument_of_perihelion(self, argument_of_perihelion):
        """
        Set the argument of perihelion of the comet object.
        """
        self.__argument_of_perihelion = argument_of_perihelion

    def get_inclination(self):
        """
        Get the inclination of the comet object.
        """
        return self.__inclination

    def set_inclination(self, inclination):
        """
        Set the inclination of the comet object.
        """
        self.__inclination = inclination

    def get_perihelion_distance(self):
        """
        Get the perihelion distance of the comet object.
        """
        return self.__perihelion_distance

    def set_perihelion_distance(self, distance):
        """
        Set the perihelion distance of the comet object.
        """
        self.__perihelion_distance = distance

    def get_eccentricity(self):
        """
        Get the eccentricity of the comet object.
        """
        return self.__eccentricity

    def set_eccentricity(self, eccentricity):
        """
        Set the eccentricity of the comet object.
        """
        self.__eccentricity = eccentricity

    def get_right_ascension(self):
        """
        Get the right ascension of the comet object.
        """
        return self.__right_ascension

    def set_right_ascension(self, right_ascension):
        """
        Set the right ascension of the comet object.
        """
        self.__right_ascension = right_ascension

    def get_declination(self):
        """
        Get the declination of the comet object.
        """
        return self.__declination

    def set_declination(self, declination):
        """
        Set the declination of the comet object.
        """
        self.__declination = declination

    def get_true_anomaly(self):
        """
        Get the true anomaly of the comet object.
        """
        return self.__true_anomaly

    def set_true_anomaly(self, true_anomaly):
        """
        Set the true anomaly of the comet object.
        """
        self.__true_anomaly = true_anomaly

    def get_geocentric_distance(self):
        """
        Get the geocentric distance of the comet object.
        """
        return self.__geocentric_distance

    def set_geocentric_distance(self, distance):
        """
        Set the geocentric distance of the comet object.
        """
        self.__geocentric_distance = distance

    def get_nucleus_position(self):
        """
        Get the nucleus position of the comet object.
        """
        return self.__nucleus_position

    def set_nucleus_position(self, position):
        """
        Set the nucleus position of the comet object.
        """
        self.__nucleus_position = position

    def get_earth_sun_distance(self):
        """
        Get the Earth-Sun distance of the comet object.
        """
        return self.__earth_sun_distance

    def set_earth_sun_distance(self, distance):
        """
        Set the Earth-Sun distance of the comet object.
        """
        self.__earth_sun_distance = distance

    def get_nucleus_coordinates(self):
        """
        Get the nucleus coordinates of the comet object.
        """
        return (self.__x1, self.__y1, self.__x2, self.__y2)

    def get_rotation_matrix(self):
        """
        Get the rotation matrix of the comet object.
        """
        return self.__orbital_to_image_rotation_matrix

    def get_kernel_rotation_matrix(self):
        """
        Get the kernel rotation matrix of the comet object.
        """
        return self.__KERNEL_ROTATION_MATRIX
