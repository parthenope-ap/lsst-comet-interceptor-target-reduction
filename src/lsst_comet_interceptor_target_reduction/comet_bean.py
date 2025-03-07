from typing import Optional


class CometBean:
    """
    A class used to represent a Comet with various orbital and positional parameters.
    """

    def __init__(self, builder: "CometBean.Builder"):
        self._kernel_rotation_matrix: tuple[int, int] = builder.kernel_rotation_matrix
        self._kepler_constant_1: Optional[float] = builder.kepler_constant_1
        self._kepler_constant_2: Optional[float] = builder.kepler_constant_2
        self._obliquity_of_the_ecliptic: Optional[float] = builder.obliquity_of_the_ecliptic
        self._anomaly: Optional[float] = builder.anomaly
        self._ascending_node: Optional[float] = builder.ascending_node
        self._argument_of_perihelion: Optional[float] = builder.argument_of_perihelion
        self._inclination: Optional[float] = builder.inclination
        self._perihelion_distance: Optional[float] = builder.perihelion_distance
        self._eccentricity: Optional[float] = builder.eccentricity
        self._right_ascension: Optional[float] = builder.right_ascension
        self._declination: Optional[float] = builder.declination
        self._true_anomaly: Optional[float] = builder.true_anomaly
        self._geocentric_distance: Optional[float] = builder.geocentric_distance
        self._nucleus_position: Optional[tuple[float, float, float]] = builder.nucleus_position
        self._earth_sun_distance: Optional[float] = builder.earth_sun_distance
        self._x1: Optional[float] = builder.x1
        self._x2: Optional[float] = builder.x2
        self._y1: Optional[float] = builder.y1
        self._y2: Optional[float] = builder.y2
        self._mod_ie: Optional[float] = builder.mod_ie

    @property
    def kernel_rotation_matrix(self) -> tuple[int, int]:
        """A 3x3 matrix for kernel rotation."""
        return self._kernel_rotation_matrix

    @kernel_rotation_matrix.setter
    def kernel_rotation_matrix(self, value: tuple[int, int]) -> None:
        self._kernel_rotation_matrix = value

    @property
    def kepler_constant_1(self) -> Optional[float]:
        """The first Kepler constant."""
        return self._kepler_constant_1

    @kepler_constant_1.setter
    def kepler_constant_1(self, value: Optional[float]) -> None:
        self._kepler_constant_1 = value

    @property
    def kepler_constant_2(self) -> Optional[float]:
        """The second Kepler constant."""
        return self._kepler_constant_2

    @kepler_constant_2.setter
    def kepler_constant_2(self, value: Optional[float]) -> None:
        self._kepler_constant_2 = value

    @property
    def obliquity_of_the_ecliptic(self) -> Optional[float]:
        """The angle between the plane of the Earth's orbit and the celestial equator."""
        return self._obliquity_of_the_ecliptic

    @obliquity_of_the_ecliptic.setter
    def obliquity_of_the_ecliptic(self, value: Optional[float]) -> None:
        self._obliquity_of_the_ecliptic = value

    @property
    def anomaly(self) -> Optional[float]:
        """The anomaly of the comet."""
        return self._anomaly

    @anomaly.setter
    def anomaly(self, value: Optional[float]) -> None:
        self._anomaly = value

    @property
    def ascending_node(self) -> Optional[float]:
        """The longitude of the ascending node."""
        return self._ascending_node

    @ascending_node.setter
    def ascending_node(self, value: Optional[float]) -> None:
        self._ascending_node = value

    @property
    def argument_of_perihelion(self) -> Optional[float]:
        """The argument of perihelion."""
        return self._argument_of_perihelion

    @argument_of_perihelion.setter
    def argument_of_perihelion(self, value: Optional[float]) -> None:
        self._argument_of_perihelion = value

    @property
    def inclination(self) -> Optional[float]:
        """The inclination of the comet's orbit."""
        return self._inclination

    @inclination.setter
    def inclination(self, value: Optional[float]) -> None:
        self._inclination = value

    @property
    def perihelion_distance(self) -> Optional[float]:
        """The distance of the comet at perihelion."""
        return self._perihelion_distance

    @perihelion_distance.setter
    def perihelion_distance(self, value: Optional[float]) -> None:
        self._perihelion_distance = value

    @property
    def eccentricity(self) -> Optional[float]:
        """The eccentricity of the comet's orbit."""
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, value: Optional[float]) -> None:
        self._eccentricity = value

    @property
    def right_ascension(self) -> Optional[float]:
        """The right ascension of the comet."""
        return self._right_ascension

    @right_ascension.setter
    def right_ascension(self, value: Optional[float]) -> None:
        self._right_ascension = value

    @property
    def declination(self) -> Optional[float]:
        """The declination of the comet."""
        return self._declination

    @declination.setter
    def declination(self, value: Optional[float]) -> None:
        self._declination = value

    @property
    def true_anomaly(self) -> Optional[float]:
        """The true anomaly of the comet."""
        return self._true_anomaly

    @true_anomaly.setter
    def true_anomaly(self, value: Optional[float]) -> None:
        self._true_anomaly = value

    @property
    def geocentric_distance(self) -> Optional[float]:
        """The distance from the Earth to the comet."""
        return self._geocentric_distance

    @geocentric_distance.setter
    def geocentric_distance(self, value: Optional[float]) -> None:
        self._geocentric_distance = value

    @property
    def nucleus_position(self) -> Optional[tuple[float, float, float]]:
        """The position of the comet's nucleus."""
        return self._nucleus_position

    @nucleus_position.setter
    def nucleus_position(self, value: Optional[tuple[float, float, float]]) -> None:
        self._nucleus_position = value

    @property
    def earth_sun_distance(self) -> Optional[float]:
        """The distance from the Earth to the Sun."""
        return self._earth_sun_distance

    @earth_sun_distance.setter
    def earth_sun_distance(self, value: Optional[float]) -> None:
        self._earth_sun_distance = value

    @property
    def x1(self) -> Optional[float]:
        """The x1 coordinate."""
        return self._x1

    @x1.setter
    def x1(self, value: Optional[float]) -> None:
        self._x1 = value

    @property
    def x2(self) -> Optional[float]:
        """The x2 coordinate."""
        return self._x2

    @x2.setter
    def x2(self, value: Optional[float]) -> None:
        self._x2 = value

    @property
    def y1(self) -> Optional[float]:
        """The y1 coordinate."""
        return self._y1

    @y1.setter
    def y1(self, value: Optional[float]) -> None:
        self._y1 = value

    @property
    def y2(self) -> Optional[float]:
        """The y2 coordinate."""
        return self._y2

    @y2.setter
    def y2(self, value: Optional[float]) -> None:
        self._y2 = value

    @property
    def mod_ie(self) -> Optional[float]:
        """The mod_ie value."""
        return self._mod_ie

    @mod_ie.setter
    def mod_ie(self, value: Optional[float]) -> None:
        self._mod_ie = value

    class Builder:
        """
        A builder class for constructing a CometBean instance.
        """

        def __init__(self):
            self.kernel_rotation_matrix: tuple[int, int] = (3, 3)
            self.kepler_constant_1: Optional[float] = None
            self.kepler_constant_2: Optional[float] = None
            self.obliquity_of_the_ecliptic: Optional[float] = None
            self.anomaly: Optional[float] = None
            self.ascending_node: Optional[float] = None
            self.argument_of_perihelion: Optional[float] = None
            self.inclination: Optional[float] = None
            self.perihelion_distance: Optional[float] = None
            self.eccentricity: Optional[float] = None
            self.right_ascension: Optional[float] = None
            self.declination: Optional[float] = None
            self.true_anomaly: Optional[float] = None
            self.geocentric_distance: Optional[float] = None
            self.nucleus_position: Optional[tuple[float, float, float]] = None
            self.earth_sun_distance: Optional[float] = None
            self.x1: Optional[float] = None
            self.x2: Optional[float] = None
            self.y1: Optional[float] = None
            self.y2: Optional[float] = None
            self.mod_ie: Optional[float] = None

        def set_kepler_constant_1(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the first Kepler constant."""
            self.kepler_constant_1 = value
            return self

        def set_kepler_constant_2(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the second Kepler constant."""
            self.kepler_constant_2 = value
            return self

        def set_obliquity_of_the_ecliptic(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the obliquity of the ecliptic."""
            self.obliquity_of_the_ecliptic = value
            return self

        def set_anomaly(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the anomaly of the comet."""
            self.anomaly = value
            return self

        def set_ascending_node(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the longitude of the ascending node."""
            self.ascending_node = value
            return self

        def set_argument_of_perihelion(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the argument of perihelion."""
            self.argument_of_perihelion = value
            return self

        def set_inclination(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the inclination of the comet's orbit."""
            self.inclination = value
            return self

        def set_perihelion_distance(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the distance of the comet at perihelion."""
            self.perihelion_distance = value
            return self

        def set_eccentricity(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the eccentricity of the comet's orbit."""
            self.eccentricity = value
            return self

        def set_right_ascension(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the right ascension of the comet."""
            self.right_ascension = value
            return self

        def set_declination(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the declination of the comet."""
            self.declination = value
            return self

        def set_true_anomaly(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the true anomaly of the comet."""
            self.true_anomaly = value
            return self

        def set_geocentric_distance(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the distance from the Earth to the comet."""
            self.geocentric_distance = value
            return self

        def set_nucleus_position(self, value: Optional[tuple[float, float, float]]) -> "CometBean.Builder":
            """Sets the position of the comet's nucleus."""
            self.nucleus_position = value
            return self

        def set_earth_sun_distance(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the distance from the Earth to the Sun."""
            self.earth_sun_distance = value
            return self

        def set_x1(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the x1 coordinate."""
            self.x1 = value
            return self

        def set_x2(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the x2 coordinate."""
            self.x2 = value
            return self

        def set_y1(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the y1 coordinate."""
            self.y1 = value
            return self

        def set_y2(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the y2 coordinate."""
            self.y2 = value
            return self

        def set_mod_ie(self, value: Optional[float]) -> "CometBean.Builder":
            """Sets the mod_ie value."""
            self.mod_ie = value
            return self

        def build(self) -> "CometBean":
            """Constructs and returns a CometBean instance."""
            return CometBean(self)
