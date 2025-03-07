import math

import numpy as np
from logger import Logger

logger = Logger()


class Comet:
    """
    A class to represent a comet and perform various computations related to its orbit and position.
    """

    def __init__(self, comet_bean):
        self.__KERNEL_ROTATION_MATRIX = (3, 3)

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

        self.__orbital_to_image_rotation_matrix = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.__ecliptic_to_equatorial_rotation_matrix = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.__temp_matrix = np.zeros(self.__KERNEL_ROTATION_MATRIX)

        # Compute image coordinates of the nucleus
        self.__compute_nucleus_coordinates()

        # Compute the rotation matrix
        self.__compute_rotation_matrix()
        logger.info("Comet object created.")

    def __compute_nucleus_coordinates(self):
        # 3600 ?
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

    def intermediate_computations(self):
        """
        Perform intermediate computations for the comet's orbit.

        Returns
        -------
            tuple: A tuple containing 'ie' and 'ac'.
        """
        # Intermediate computations
        ie = int(self.__mod_ie * self.__true_anomaly)
        ac = math.tan(0.5 * np.deg2rad(self.__true_anomaly))
        if ac < 0.0:
            ie -= 1
        return ie, ac

    def __compute_rotation_matrix(self):
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

    @property
    def header(self):
        """Get the header."""
        return self.__header

    @header.setter
    def header(self, header):
        """Set the header."""
        self.__header = header

    @property
    def time_array(self):
        """Get the time array."""
        return self.__time_array

    @time_array.setter
    def time_array(self, time_array):
        """Set the time array."""
        self.__time_array = time_array

    @property
    def orbital_to_image_rotation_matrix(self):
        """Get the orbital to image rotation matrix."""
        return self.__orbital_to_image_rotation_matrix

    @orbital_to_image_rotation_matrix.setter
    def orbital_to_image_rotation_matrix(self, matrix):
        """Set the orbital to image rotation matrix."""
        self.__orbital_to_image_rotation_matrix = matrix

    @property
    def ecliptic_to_equatorial_rotation_matrix(self):
        """Get the ecliptic to equatorial rotation matrix."""
        return self.__ecliptic_to_equatorial_rotation_matrix

    @ecliptic_to_equatorial_rotation_matrix.setter
    def ecliptic_to_equatorial_rotation_matrix(self, matrix):
        """Set the ecliptic to equatorial rotation matrix."""
        self.__ecliptic_to_equatorial_rotation_matrix = matrix

    @property
    def temp_matrix(self):
        """Get the temporary matrix."""
        return self.__temp_matrix

    @temp_matrix.setter
    def temp_matrix(self, matrix):
        """Set the temporary matrix."""
        self.__temp_matrix = matrix

    @property
    def base_image(self):
        """Get the base image."""
        return self.__base_image

    @base_image.setter
    def base_image(self, image):
        """Set the base image."""
        self.__base_image = image

    @property
    def model_image(self):
        """Get the model image."""
        return self.__model_image

    @model_image.setter
    def model_image(self, image):
        """Set the model image."""
        self.__model_image = image

    @property
    def extended_image(self):
        """Get the extended image."""
        return self.__extended_image

    @extended_image.setter
    def extended_image(self, image):
        """Set the extended image."""
        self.__extended_image = image

    @property
    def comet_image(self):
        """Get the comet image."""
        return self.__comet_image

    @comet_image.setter
    def comet_image(self, image):
        """Set the comet image."""
        self.__comet_image = image

    @property
    def kepler_constant_1(self):
        """Get the first Kepler constant."""
        return self.__kepler_constant_1

    @property
    def kepler_constant_2(self):
        """Get the second Kepler constant."""
        return self.__kepler_constant_2

    @property
    def obliquity_of_the_ecliptic(self):
        """Get the obliquity of the ecliptic."""
        return self.__obliquity_of_the_ecliptic

    @property
    def anomaly(self):
        """Get the anomaly."""
        return self.__anomaly

    @anomaly.setter
    def anomaly(self, anomaly):
        """Set the anomaly."""
        self.__anomaly = anomaly

    @property
    def ascending_node(self):
        """Get the ascending node."""
        return self.__ascending_node

    @ascending_node.setter
    def ascending_node(self, ascending_node):
        """Set the ascending node."""
        self.__ascending_node = ascending_node

    @property
    def argument_of_perihelion(self):
        """Get the argument of perihelion."""
        return self.__argument_of_perihelion

    @argument_of_perihelion.setter
    def argument_of_perihelion(self, argument_of_perihelion):
        """Set the argument of perihelion."""
        self.__argument_of_perihelion = argument_of_perihelion

    @property
    def inclination(self):
        """Get the inclination."""
        return self.__inclination

    @inclination.setter
    def inclination(self, inclination):
        """Set the inclination."""
        self.__inclination = inclination

    @property
    def perihelion_distance(self):
        """Get the perihelion distance."""
        return self.__perihelion_distance

    @perihelion_distance.setter
    def perihelion_distance(self, distance):
        """Set the perihelion distance."""
        self.__perihelion_distance = distance

    @property
    def eccentricity(self):
        """Get the eccentricity."""
        return self.__eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        """Set the eccentricity."""
        self.__eccentricity = eccentricity

    @property
    def right_ascension(self):
        """Get the right ascension."""
        return self.__right_ascension

    @right_ascension.setter
    def right_ascension(self, right_ascension):
        """Set the right ascension."""
        self.__right_ascension = right_ascension

    @property
    def declination(self):
        """Get the declination."""
        return self.__declination

    @declination.setter
    def declination(self, declination):
        """Set the declination."""
        self.__declination = declination

    @property
    def true_anomaly(self):
        """Get the true anomaly."""
        return self.__true_anomaly

    @true_anomaly.setter
    def true_anomaly(self, true_anomaly):
        """Set the true anomaly."""
        self.__true_anomaly = true_anomaly

    @property
    def geocentric_distance(self):
        """Get the geocentric distance."""
        return self.__geocentric_distance

    @geocentric_distance.setter
    def geocentric_distance(self, distance):
        """Set the geocentric distance."""
        self.__geocentric_distance = distance

    @property
    def nucleus_position(self):
        """Get the nucleus position."""
        return self.__nucleus_position

    @nucleus_position.setter
    def nucleus_position(self, position):
        """Set the nucleus position."""
        self.__nucleus_position = position

    @property
    def earth_sun_distance(self):
        """Get the Earth-Sun distance."""
        return self.__earth_sun_distance

    @earth_sun_distance.setter
    def earth_sun_distance(self, distance):
        """Set the Earth-Sun distance."""
        self.__earth_sun_distance = distance

    @property
    def nucleus_coordinates(self):
        """Get the nucleus coordinates."""
        return (self.__x1, self.__y1, self.__x2, self.__y2)

    @property
    def rotation_matrix(self):
        """Get the rotation matrix."""
        return self.__orbital_to_image_rotation_matrix

    @property
    def kernel_rotation_matrix(self):
        """Get the kernel rotation matrix."""
        return self.__KERNEL_ROTATION_MATRIX
