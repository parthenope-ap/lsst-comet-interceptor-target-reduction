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
from .comet_bean import CometBean

# Define a list of required constants and load them
CONSTANTS = [
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


class CometBeanDAO:
    """
    Data Access Object (DAO) for creating a CometBean object.
    Dynamically loads constants from a specified module for the given comet.
    """

    def __init__(self, comet_name: str) -> None:
        """
        Initializes the DAO with the name of the comet.

        Parameters
        ----------
        comet_name : str
            The name of the comet to load constants for.
        """
        self.comet_name = comet_name

    def create_comet_bean(self, constants: dict) -> CometBean:
        """
        Creates a CometBean object using constants loaded from the specified module.

        Parameters
        ----------
        constants : dict
            A dictionary containing the loaded constants for the comet.
        Returns
        -------
        CometBean
            A CometBean object containing the loaded constants.
        """

        # Create a CometBean object using the Builder pattern
        comet_bean = (
            CometBean.Builder()
            .set_kepler_constant_1(constants["KEPLER_CONSTANT_1"])
            .set_kepler_constant_2(constants["KEPLER_CONSTANT_2"])
            .set_obliquity_of_the_ecliptic(constants["OBLIQUITY_OF_THE_ECLIPTIC"])
            .set_anomaly(constants["ANOMALY"])
            .set_ascending_node(constants["ASCENDING_NODE"])
            .set_argument_of_perihelion(constants["ARGUMENT_OF_PERIHELION"])
            .set_inclination(constants["INCLINATION"])
            .set_perihelion_distance(constants["PERIHELION_DISTANCE"])
            .set_eccentricity(constants["ECCENTRICITY"])
            .set_right_ascension(constants["RIGHT_ASCENSION"])
            .set_declination(constants["DECLINATION"])
            .set_true_anomaly(constants["TRUE_ANOMALY"])
            .set_geocentric_distance(constants["GEOCENTRIC_DISTANCE"])
            .set_nucleus_position(constants["NUCLUES_POSITION"])
            .set_earth_sun_distance(constants["EARTH_SUN_DISTANCE"])
            .set_x1(constants["X1"])
            .set_y1(constants["Y1"])
            .set_x2(constants["X2"])
            .set_y2(constants["Y2"])
            .set_mod_ie(constants["MOD_IE"])
            .build()
        )

        return comet_bean
