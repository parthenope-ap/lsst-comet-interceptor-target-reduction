import math

import numpy as np
import pandas as pd

from .comet_bean import CometBean
from .plotter import check_comet_portion_size


def scoring_fn(
    comet_image: pd.DataFrame,
    montecarlo_simulation: np.ndarray,
    matrix_size: int,
    portion_comet: tuple,
    sky_background_value: float,
    threshold_min: float,
    perihelion_distance: float,
    eccentricity: float,
    multiplier: float,
    anomaly: float,
    binning: int = 1,
    minimum_size: int = 100,
):
    """
    Scoring function for the Monte Carlo simulation.

    Parameters
    ----------
    comet_image : pd.DataFrame
        Dataframe containing the comet image data.
    montecarlo_simulation : np.ndarray
        The Monte Carlo simulation data.
    matrix_size : int
        The size of the matrix.
    portion_comet : tuple
        The portion of the comet image.
    sky_background_value : float
        The sky background value.
    threshold_min : float
        The minimum threshold value.
    perihelion_distance : float
        The perihelion distance.
    eccentricity : float
        The eccentricity.
    multiplier : float
        The multiplier value.
    anomaly : float
        The anomaly value.
    binning : int
        The binning value.
    minimum_size : int
        The minimum size value.
    """

    comet = np.zeros((matrix_size, matrix_size))
    max_montecarlo_value = 0.0
    de = 0.0
    fit_error = 0.0
    row = col = matrix_size
    max_montecarlo_value = max(max(row) for row in montecarlo_simulation)

    # Check and adjust the portion if necessary
    portion_comet = check_comet_portion_size(comet_image, portion_comet, minimum_size)

    # TODO: identify automatically the portion of the comet image to use
    comet_portion = comet_image.iloc[portion_comet[0] : portion_comet[1], portion_comet[2] : portion_comet[3]]
    for i in range(10):
        for j in range(10):
            de += 0.01 * comet_portion.iloc[i, j]

    for i in range(matrix_size):
        for j in range(matrix_size):
            for k in range(binning):
                for n in range(binning):
                    comet[i, j] += comet_portion.iloc[binning * i + k, binning * j + n]
            # TODO: check the division
            comet[i, j] /= np.double(binning**2)

    for i in range(row):
        for j in range(col):
            if comet[i, j] <= np.double(sky_background_value):
                comet[i, j] = 0.0
            if comet[i, j] > np.double(sky_background_value):
                comet[i, j] -= np.double(sky_background_value)
            am = np.double(multiplier) * montecarlo_simulation[i, j] / max_montecarlo_value - comet[i, j]
            om = np.double(multiplier) * montecarlo_simulation[i, j] / max_montecarlo_value + comet[i, j]

            num = np.double(multiplier) * montecarlo_simulation[i, j] / max_montecarlo_value
            if num > np.double(threshold_min) and comet[i, j] > np.double(threshold_min):
                fit_error += am**2 / (om**2)
    result = (
        perihelion_distance * (1.0 + eccentricity) / (1.0 + eccentricity * math.cos(0.1 * math.pi * anomaly))
    )
    fit_error = 2.0 * fit_error / (matrix_size * matrix_size)

    return [max_montecarlo_value, comet, montecarlo_simulation, result, de, fit_error]


class MonteCarlo:
    """
    Monte Carlo simulation class for comet dust tail simulation.
    """

    def __init__(
        self,
        comet: CometBean,
        dim_matrix_image: int,
        num_iter: int,
        depend: bool = False,
        t_size: int = 3600,
        param_00: float = 0.05,
        param_01: float = 360.0,
        param_02: float = 0.9,
        param_03: float = 1.0,
        param_3: float = 0.009,
        exp_val: float = -2.0,
        k: float = 1.0,
    ):
        """
        Initializes the Monte Carlo simulation object with the given comet and parameters.

        Parameters
        ----------
        comet : CometBean
            The comet object containing the necessary parameters.
        dim_matrix_image : int
            The dimension of the image matrix.
        num_iter : int
            The number of iterations for the Monte Carlo simulation.
        depend : bool
            A flag indicating whether the Monte Carlo simulation depends on the comet's position.
        t_size : int
            The size of the temporary array.
        param_00 : float
            The parameter for the Monte Carlo simulation.
        param_01 : float
            The parameter for the Monte Carlo simulation.
        param_02 : float
            The parameter for the Monte Carlo simulation.
        param_03 : float
            The parameter for the Monte Carlo simulation.
        param_3 : float
            The parameter for the Monte Carlo simulation.
        exp_val : float
            The exponent value for the Monte Carlo simulation.
        k : float
            The k parameter for the Monte Carlo simulation.
        """

        self.comet = comet
        self.__KERNEL_ROTATION_MATRIX = comet.get_kernel_rotation_matrix()
        self.nc = dim_matrix_image
        self.num_iter_montecarlo = num_iter
        self.t = np.zeros(t_size)
        self.q = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.r = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.z = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.f = np.zeros((self.nc, self.nc))
        self.h = np.zeros((1600, 1600), dtype=int)
        self.x1 = comet.get_nucleus_coordinates()[0]
        self.x2 = comet.get_nucleus_coordinates()[2]
        self.y1 = comet.get_nucleus_coordinates()[1]
        self.y2 = comet.get_nucleus_coordinates()[3]
        self.r = comet.get_orbital_to_image_rotation_matrix()
        self.q = comet.get_ecliptic_to_equatorial_rotation_matrix()
        self.z = comet.get_temp_matrix()
        self.depend = depend
        self.param_00 = param_00
        self.param_01 = param_01
        self.param_02 = param_02
        self.param_03 = param_03
        self.param_3 = param_3
        self.exp_val = exp_val
        self.k = k
        self.PARAM_1 = 180.0  # Costant
        self.PARAM_2 = 0.5  # Costant

    def fit(self) -> np.ndarray:
        """
        Fit method for Monte Carlo simulation.

        Returns
        -------
        np.array
            The Monte Carlo simulation data.
        """
        # logger.info("Monte Carlo simulation...")
        comet = self.comet
        g1 = comet.get_kepler_constant_1()
        g2 = comet.get_kepler_constant_2()
        qc = comet.get_perihelion_distance()
        ec = comet.get_eccentricity()
        ia = comet.get_anomaly()
        ie, ac = comet.intermediate_computations()
        pig = np.double(math.atan(1.0) / 45.0)
        nc = self.nc

        # logger.warning(f"ie: {ie}, ac: {ac}")
        # logger.warning(f"ia: {ia}")
        if ec == 1.0:
            y = g1 * math.sqrt(2.0 * qc**3)
            for i in range(ie - ia + 1):
                u = math.tan(self.param_00 * pig * (ia + i))
                self.t[i] = y * (u + u**3 / 3.0)
            tc = y * (ac + ac**3 / 3.0)
        elif ec > 1.0:
            x = qc / (ec - 1.0)
            y = 2.0 * g1 * math.sqrt(x**3)
            for i in range(ie - ia + 1):
                u = math.sqrt((ec - 1.0) / (ec + 1.0)) * math.tan(self.param_00 * pig * (ia + i))
                self.t[i] = y * (0.5 * math.log((1.0 - u) / (1.0 + u)) + ec * u / (1.0 - u**2))
            u = ac * math.sqrt((ec - 1.0) / (ec + 1.0))
            tc = y * (0.5 * math.log((1.0 - u) / (1.0 + u)) + ec * u / (1.0 - u**2))
        elif ec < 1.0:
            x = qc / (1.0 - ec)
            y = 2.0 * g1 * math.sqrt(x**3)
            for i in range(ie - ia + 1):
                u = math.sqrt((1.0 - ec) / (1.0 + ec)) * math.tan(self.param_00 * pig * (ia + i))
                self.t[i] = y * (math.atan(u) - ec * u / (1.0 + u**2))
            u = ac * math.sqrt((1.0 - ec) / (1.0 + ec))
            tc = y * (math.atan(u) - ec * u / (1.0 + u**2))

        w = ac**2 + 1.0
        sd = 2.0 * ac / w
        cd = (2.0 - w) / w
        rc = qc * (1.0 + ec) / (1.0 + ec * cd)
        xc = rc * cd
        yc = rc * sd
        vc = math.sqrt(g2 / (qc * (1.0 + ec)))

        for i in range(ie - ia):
            am = self.param_03 * pig * (i + ia)
            xi = math.cos(am)
            ep = math.sin(am)
            rs = qc * (1.0 + ec) / (1.0 + ec * xi)
            vx = vc * ec * ep
            vy = vc * (1.0 + ec * xi)

            for _ in range(self.num_iter_montecarlo):
                ra = pig * self.param_01 * (np.random.random() - 0.5)
                de = pig * self.PARAM_1 * (np.random.random() - self.PARAM_2)
                am = self.param_3 / rs
                um = 1.0 - self.param_02 * np.random.random()
                x = math.cos(de)
                vdx = vx - am * x * math.cos(ra)
                vdy = vy - am * x * math.sin(ra)
                vdz = am * math.sin(de)
                de = x
                ad = vdy**2 + vdz**2
                rd = ad + vdx**2
                av = g2 * um
                bv = rs**2 * ad / av
                ed = math.sqrt(bv * (rd / av - 2.0 / rs) + 1.0)
                ad = math.sqrt(ad)
                av = math.sqrt(bv / (g2 * um * ed**2))
                sn = vdx * av
                cn = ad * av - 1.0 / ed
                qd = bv / (ed + 1.0)
                om = sn / (cn + 1.0)
                vdx = vdz / ad
                vdy = vdy / ad

                if 0.99999 <= ed <= 1.00001:
                    u = 1.0
                    v = om
                    w = ac
                    am = g1 * math.sqrt(2.0 * qd**3 / um)
                    x = am * (v + v**3 / 3.0) + tc - self.t[i]
                    y = am * (w + w**3 / 3.0)
                    ra = abs(x - y)
                    while ra > 1.0e-6:
                        w += (x - y) / (am * (1.0 + w**2))
                        y = am * (w + w**3 / 3.0)
                        ra = abs(x - y)

                elif ed > 1.00001:
                    u = math.sqrt((ed - 1.0) / (ed + 1.0))
                    v = u * om
                    w = u * ac
                    am = qd / (ed - 1.0)
                    am = 2.0 * g1 * math.sqrt(am**3 / um)
                    x = am * (0.5 * math.log((1.0 - v) / (1.0 + v)) + ed * v / (1.0 - v**2)) + tc - self.t[i]
                    y = am * (0.5 * math.log((1.0 - w) / (1.0 + w)) + ed * w / (1.0 - w**2))
                    v = 1.0 - w**2
                    ra = abs(x - y)
                    while ra > 1.0e-6:
                        w += (y - x) * v**2 / (am * (ed * (v - 2.0) + v))
                        v = 1.0 - w**2
                        y = am * (0.5 * math.log((1.0 - w) / (1.0 + w)) + ed * w / v)
                        ra = abs(x - y)

                elif ed < 0.99999:
                    u = math.sqrt((1.0 - ed) / (1.0 + ed))
                    v = u * om
                    w = u * ac
                    am = qd / (1.0 - ed)
                    am = 2.0 * g1 * math.sqrt(am**3 / um)
                    x = am * (math.atan(v) - ed * v / (1.0 + v**2)) + tc - self.t[i]
                    y = am * (math.atan(w) - ed * w / (1.0 + w**2))
                    v = 1.0 + w**2
                    ra = abs(x - y)
                    while ra > 1.0e-6:
                        w += (x - y) * v**2 / (am * (ed * (v - 2.0) + v))
                        v = 1.0 + w**2
                        y = am * (math.atan(w) - ed * w / v)
                        ra = abs(x - y)

                ad = w / u
                om = ad**2 + 1.0
                sd = 2.0 * ad / om
                cd = (2.0 - om) / om
                rd = bv / (ed * cd + 1.0)
                av = cd * cn + sd * sn
                bv = sd * cn - cd * sn
                self.z[0, 0] = 149.6 * (rd * (xi * av - ep * bv * vdy) - xc)
                self.z[1, 0] = 149.6 * (rd * (ep * av + xi * bv * vdy) - yc)
                self.z[2, 0] = 149.6 * rd * bv * vdx

                for j in range(self.__KERNEL_ROTATION_MATRIX[0]):
                    self.q[j, 0] = 0.0
                    for n in range(self.__KERNEL_ROTATION_MATRIX[1]):
                        self.q[j, 0] += self.r[j, n] * self.z[n, 0]

                if self.x1 < self.q[1, 0] < self.x2 and self.y1 < self.q[2, 0] < self.y2:
                    j = int((self.q[2, 0] - self.y1) * nc / (self.y2 - self.y1))
                    n = int((self.q[1, 0] - self.x1) * nc / (self.x2 - self.x1))
                    x = math.log(2000.0 * (1.0 - um))

                    if self.depend:
                        y = math.exp(-0.2 * x**2)
                        if rs <= 3.8:
                            self.f[n, j] += (self.t[i + 1] - self.t[i]) * y * de / rs
                        if rs > 3.8:
                            self.f[n, j] += (self.t[i + 1] - self.t[i]) * y * de * 3.8 / (rs**2)
                    else:
                        y = math.exp(self.exp_val * x**2)
                        self.f[n, j] += (self.t[i + 1] - self.t[i]) * y * de / rs * self.k

        return self.f
