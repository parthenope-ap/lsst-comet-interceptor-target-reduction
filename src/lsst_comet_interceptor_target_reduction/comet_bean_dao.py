# import importlib

# from comet_bean import CometBean

# # DAO Class
# class CometBeanDAO:
#     """
#     Data Access Object (DAO) for accessing and managing CometBean data.
#     This class abstracts database interactions to fetch comet data and construct CometBean objects.
#     """
#     def __init__(self, comet_name):
#         """
#         Initializes the DAO. Connects to the database if a database file is provided.
#         """
#         self.comet_name = comet_name
#         constants_module = importlib.import_module(f"input.{self.comet_name}.{self.comet_name}")
#         KEPLER_CONSTANT_1 = getattr(constants_module, "KEPLER_CONSTANT_1", None)
#         KEPLER_CONSTANT_2 = getattr(constants_module, "KEPLER_CONSTANT_2", None)
#         OBLIQUITY_OF_THE_ECLIPTIC = getattr(constants_module, "OBLIQUITY_OF_THE_ECLIPTIC", None)
#         ANOMALY = getattr(constants_module, "ANOMALY", None)
#         ASCENDING_NODE = getattr(constants_module, "ASCENDING_NODE", None)
#         ARGUMENT_OF_PERIHELION = getattr(constants_module, "ARGUMENT_OF_PERIHELION", None)
#         INCLINATION = getattr(constants_module, "INCLINATION", None)
#         PERIHELION_DISTANCE = getattr(constants_module, "PERIHELION_DISTANCE", None)
#         ECCENTRICITY = getattr(constants_module, "ECCENTRICITY", None)
#         RIGHT_ASCENSION = getattr(constants_module, "RIGHT_ASCENSION", None)
#         DECLINATION = getattr(constants_module, "DECLINATION", None)
#         TRUE_ANOMALY = getattr(constants_module, "TRUE_ANOMALY", None)
#         GEOCENTRIC_DISTANCE = getattr(constants_module, "GEOCENTRIC_DISTANCE", None)
#         NUCLUES_POSITION = getattr(constants_module, "NUCLUES_POSITION", None)
#         EARTH_SUN_DISTANCE = getattr(constants_module, "EARTH_SUN_DISTANCE", None)
#         X1 = getattr(constants_module, "X1", None)
#         Y1 = getattr(constants_module, "Y1", None)
#         X2 = getattr(constants_module, "X2", None)
#         Y2 = getattr(constants_module, "Y2", None)
#         MOD_IE = getattr(constants_module, "MOD_IE", None)
#         # Create a CometBean object using the Builder
#         comet_bean = (CometBean.Builder()
#                     .set_kepler_constant_1(KEPLER_CONSTANT_1)
#                     .set_kepler_constant_2(KEPLER_CONSTANT_2)
#                     .set_obliquity_of_the_ecliptic(OBLIQUITY_OF_THE_ECLIPTIC)
#                     .set_anomaly(ANOMALY)
#                     .set_ascending_node(ASCENDING_NODE)
#                     .set_argument_of_perihelion(ARGUMENT_OF_PERIHELION)
#                     .set_inclination(INCLINATION)
#                     .set_perihelion_distance(PERIHELION_DISTANCE)
#                     .set_eccentricity(ECCENTRICITY)
#                     .set_right_ascension(RIGHT_ASCENSION)
#                     .set_declination(DECLINATION)
#                     .set_true_anomaly(TRUE_ANOMALY)
#                     .set_geocentric_distance(GEOCENTRIC_DISTANCE)
#                     .set_nucleus_position(NUCLUES_POSITION)
#                     .set_earth_sun_distance(EARTH_SUN_DISTANCE)
#                     .set_x1(X1)
#                     .set_y1(Y1)
#                     .set_x2(X2)
#                     .set_y2(Y2)
#                     .set_mod_ie(MOD_IE)
#                     .build())
#         return comet_bean
import importlib

from .comet_bean import CometBean


class CometBeanDAO:
    """
    Data Access Object (DAO) for creating a CometBean object.
    Dynamically loads constants from a specified module for the given comet.
    """

    def __init__(self, comet_name: str):
        """
        Initializes the DAO with the given comet name.

        :param comet_name: The name of the comet (used to dynamically load constants).
        """
        self.comet_name = comet_name

    def create_comet_bean(self) -> CometBean:
        """
        Creates and returns a CometBean object using constants dynamically loaded from a module.

        :return: A CometBean object.
        """
        # Dynamically import the module containing constants
        try:
            constants_module = importlib.import_module(f"input.{self.comet_name}.{self.comet_name}")
        except ModuleNotFoundError as e:
            raise ImportError(f"Could not find module for comet '{self.comet_name}'.") from e

        # Define a list of required constants and load them
        constants = [
            "KEPLER_CONSTANT_1",
            "KEPLER_CONSTANT_2",
            "OBLIQUITY_OF_THE_ECLIPTIC",
            "ANOMALY",
            "ASCENDING_NODE",
            "ARGUMENT_OF_PERIHELION",
            "INCLINATION",
            "PERIHELION_DISTANCE",
            "ECCENTRICITY",
            "RIGHT_ASCENSION",
            "DECLINATION",
            "TRUE_ANOMALY",
            "GEOCENTRIC_DISTANCE",
            "NUCLUES_POSITION",
            "EARTH_SUN_DISTANCE",
            "X1",
            "Y1",
            "X2",
            "Y2",
            "MOD_IE",
        ]

        # Load constants using getattr, defaulting to None if not found
        loaded_constants = {const: getattr(constants_module, const, None) for const in constants}

        # Create a CometBean object using the Builder pattern
        comet_bean = (
            CometBean.Builder()
            .set_kepler_constant_1(loaded_constants["KEPLER_CONSTANT_1"])
            .set_kepler_constant_2(loaded_constants["KEPLER_CONSTANT_2"])
            .set_obliquity_of_the_ecliptic(loaded_constants["OBLIQUITY_OF_THE_ECLIPTIC"])
            .set_anomaly(loaded_constants["ANOMALY"])
            .set_ascending_node(loaded_constants["ASCENDING_NODE"])
            .set_argument_of_perihelion(loaded_constants["ARGUMENT_OF_PERIHELION"])
            .set_inclination(loaded_constants["INCLINATION"])
            .set_perihelion_distance(loaded_constants["PERIHELION_DISTANCE"])
            .set_eccentricity(loaded_constants["ECCENTRICITY"])
            .set_right_ascension(loaded_constants["RIGHT_ASCENSION"])
            .set_declination(loaded_constants["DECLINATION"])
            .set_true_anomaly(loaded_constants["TRUE_ANOMALY"])
            .set_geocentric_distance(loaded_constants["GEOCENTRIC_DISTANCE"])
            .set_nucleus_position(loaded_constants["NUCLUES_POSITION"])
            .set_earth_sun_distance(loaded_constants["EARTH_SUN_DISTANCE"])
            .set_x1(loaded_constants["X1"])
            .set_y1(loaded_constants["Y1"])
            .set_x2(loaded_constants["X2"])
            .set_y2(loaded_constants["Y2"])
            .set_mod_ie(loaded_constants["MOD_IE"])
            .build()
        )

        return comet_bean
