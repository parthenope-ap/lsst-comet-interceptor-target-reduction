import math

import numpy as np
import pandas as pd

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
    Scoring function used for evaluating the fit error of the Monte Carlo simulations.

    Parameters
    ----------
    comet_image : pd.DataFrame
        DataFrame containing the comet image data.
    montecarlo_simulation : np.ndarray
        2D array of Monte Carlo simulation data.
    matrix_size : int
        Size of the plot grid.
    comet : np.ndarray
        2D array representing the comet image data.
    portion_comet : tuple[int, int, int, int]
        The portion defined as (start_row, end_row, start_col, end_col).
    sky_background_value : float
        Value of the sky background to be subtracted from the comet data.
    threshold_min : float
        Minimum threshold value for the comet and Monte Carlo data.
    perihelion_distance : float
        Perihelion distance of the comet.
    eccentricity : float
        Eccentricity of the comet's orbit.
    multiplier : float
        Multiplier for scaling the simulation data.
    anomaly : float
        Anomaly value for the comet's orbit.
    binning : int, optional
        Binning factor for the comet data (default is 1).
    minimum_size : int, optional
        Minimum required size for the portion (default is 100).

    Returns
    -------
    list
        List containing the maximum Monte Carlo value, comet data, and Monte Carlo simulation data.
    """

    comet = np.zeros((matrix_size, matrix_size))
    max_montecarlo_value = 0.0
    de = 0.0
    fit_error = 0.0
    row = col = matrix_size
    max_montecarlo_value = max(max(row) for row in montecarlo_simulation)

    # Check and adjust the portion if necessary
    portion_comet = check_comet_portion_size(comet_image, portion_comet, minimum_size)
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
            if comet[i, j] > np.double(sky_background_value):
                comet[i, j] -= np.double(sky_background_value)
            am = np.double(multiplier) * montecarlo_simulation[i, j] / max_montecarlo_value - comet[i, j]
            om = np.double(multiplier) * montecarlo_simulation[i, j] / max_montecarlo_value + comet[i, j]
            montecarlo_value = np.double(multiplier) * montecarlo_simulation[i, j] / max_montecarlo_value
            if montecarlo_value > np.double(threshold_min) and comet[i, j] > np.double(threshold_min):
                fit_error += am**2 / (om**2)
    result = (
        perihelion_distance * (1.0 + eccentricity) / (1.0 + eccentricity * math.cos(0.1 * math.pi * anomaly))
    )
    fit_error = 2.0 * fit_error / (matrix_size * matrix_size)
    return [max_montecarlo_value, comet, montecarlo_simulation, result, de, fit_error]


class MonteCarlo:
    """
    Monte Carlo simulation class for comet data analysis.

    This class performs Monte Carlo simulations to fit comet data and compute various parameters
    related to the comet's orbit and characteristics.

    Attributes
    ----------
    comet : np.ndarray
        Comet data.
    dim_matrix_image : int
        Dimension of the matrix image.
    num_iter : int
        Number of iterations for the Monte Carlo simulation.
    depend : bool, optional
        Dependency flag (default is False).
    t_size : int, optional
        Size of the time array (default is 3600).
    param_00 : float, optional
        Parameter 00 for the simulation (default is 0.05).
    param_01 : float, optional
        Parameter 01 for the simulation (default is 360.0).
    param_02 : float, optional
        Parameter 02 for the simulation (default is 0.9).
    param_03 : float, optional
        Parameter 03 for the simulation (default is 1.0).
    param_3 : float, optional
        Parameter 3 for the simulation (default is 0.009).
    exp_val : float, optional
        Exponential value for the simulation (default is -2.0).
    k : int, optional
        Constant for the simulation (default is 1).
    """

    def __init__(
        self,
        comet: np.ndarray,
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
        k: int = 1,
    ):
        self.comet = comet
        self.__KERNEL_ROTATION_MATRIX = comet.kernel_rotation_matrix
        self.nc = dim_matrix_image
        self.num_iter_montecarlo = num_iter
        self.t = np.zeros(t_size)
        self.q = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.r = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.z = np.zeros(self.__KERNEL_ROTATION_MATRIX)
        self.f = np.zeros((self.nc, self.nc))
        self.h = np.zeros((1600, 1600), dtype=int)
        self.x1, self.x2, self.y1, self.y2 = comet.nucleus_coordinates
        self.r = comet.orbital_to_image_rotation_matrix
        self.q = comet.ecliptic_to_equatorial_rotation_matrix
        self.z = comet.temp_matrix
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
        Perform the Monte Carlo simulation to fit the comet data.

        Returns
        -------
        np.ndarray
            2D array representing the fitted comet data.
        """
        comet = self.comet
        g1 = comet.kepler_constant_1
        g2 = comet.kepler_constant_2
        qc = comet.perihelion_distance
        ec = comet.eccentricity
        ia = comet.anomaly
        ie, ac = comet.perform_intermediate_computations()
        pig = np.double(math.atan(1.0) / 45.0)
        nc = self.nc

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
