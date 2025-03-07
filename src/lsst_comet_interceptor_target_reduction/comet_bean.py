class CometBean:
    """
    A class used to represent a Comet with various orbital and positional parameters.
    """

    def __init__(self, builder):
        self.kernel_rotation_matrix = builder.kernel_rotation_matrix
        self.kepler_constant_1 = builder.kepler_constant_1
        self.kepler_constant_2 = builder.kepler_constant_2
        self.obliquity_of_the_ecliptic = builder.obliquity_of_the_ecliptic
        self.anomaly = builder.anomaly
        self.ascending_node = builder.ascending_node
        self.argument_of_perihelion = builder.argument_of_perihelion
        self.inclination = builder.inclination
        self.perihelion_distance = builder.perihelion_distance
        self.eccentricity = builder.eccentricity
        self.right_ascension = builder.right_ascension
        self.declination = builder.declination
        self.true_anomaly = builder.true_anomaly
        self.geocentric_distance = builder.geocentric_distance
        self.nucleus_position = builder.nucleus_position
        self.earth_sun_distance = builder.earth_sun_distance
        self.x1 = builder.x1
        self.x2 = builder.x2
        self.y1 = builder.y1
        self.y2 = builder.y2
        self.mod_ie = builder.mod_ie

    class Builder:
        """
        A class used to build a CometBean object.
        """

        def __init__(self):
            self.kernel_rotation_matrix = (3, 3)
            self.kepler_constant_1 = None
            self.kepler_constant_2 = None
            self.obliquity_of_the_ecliptic = None
            self.anomaly = None
            self.ascending_node = None
            self.argument_of_perihelion = None
            self.inclination = None
            self.perihelion_distance = None
            self.eccentricity = None
            self.right_ascension = None
            self.declination = None
            self.true_anomaly = None
            self.geocentric_distance = None
            self.nucleus_position = None
            self.earth_sun_distance = None
            self.x1 = None
            self.x2 = None
            self.y1 = None
            self.y2 = None

        def set_kepler_constant_1(self, value):
            """Set the first Kepler constant."""
            self.kepler_constant_1 = value
            return self

        def set_kepler_constant_2(self, value):
            """Set the second Kepler constant."""
            self.kepler_constant_2 = value
            return self

        def set_obliquity_of_the_ecliptic(self, value):
            """Set the obliquity of the ecliptic."""
            self.obliquity_of_the_ecliptic = value
            return self

        def set_anomaly(self, value):
            """Set the anomaly."""
            self.anomaly = value
            return self

        def set_ascending_node(self, value):
            """Set the ascending node."""
            self.ascending_node = value
            return self

        def set_argument_of_perihelion(self, value):
            """Set the argument of perihelion."""
            self.argument_of_perihelion = value
            return self

        def set_inclination(self, value):
            """Set the inclination."""
            self.inclination = value
            return self

        def set_perihelion_distance(self, value):
            """Set the perihelion distance."""
            self.perihelion_distance = value
            return self

        def set_eccentricity(self, value):
            """Set the eccentricity."""
            self.eccentricity = value
            return self

        def set_right_ascension(self, value):
            """Set the right ascension."""
            self.right_ascension = value
            return self

        def set_declination(self, value):
            """Set the declination."""
            self.declination = value
            return self

        def set_true_anomaly(self, value):
            """Set the true anomaly."""
            self.true_anomaly = value
            return self

        def set_geocentric_distance(self, value):
            """Set the geocentric distance."""
            self.geocentric_distance = value
            return self

        def set_nucleus_position(self, value):
            """Set the nucleus position."""
            self.nucleus_position = value
            return self

        def set_earth_sun_distance(self, value):
            """Set the Earth-Sun distance."""
            self.earth_sun_distance = value
            return self

        def set_x1(self, value):
            """Set the x1 coordinate."""
            self.x1 = value
            return self

        def set_x2(self, value):
            """Set the x2 coordinate."""
            self.x2 = value
            return self

        def set_y1(self, value):
            """Set the y1 coordinate."""
            self.y1 = value
            return self

        def set_y2(self, value):
            """Set the y2 coordinate."""
            self.y2 = value
            return self

        def set_mod_ie(self, value):
            """Set the mod IE."""
            self.mod_ie = value
            return self

        def build(self):
            """Build the CometBean object."""
            return CometBean(self)
