import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logger import Logger

logger = Logger()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between a minimum and maximum value.

    Parameters
    ----------
    value : float
        The value to be clamped.
    min_val : float
        The minimum value to clamp to.
    max_val : float
        The maximum value to clamp to.

    Returns
    -------
    float
        The clamped value.
    """
    return max(min(value, max_val), min_val)


def plot_isofote(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    comet: np.ndarray,
    montecarlo_simulation: np.ndarray,
    max_montecarlo_value: float,
    plot_multiplier: float,
    om: float,
    am: float,
    xi: float,
    multiplier: float,
    levels: list,
    ra_constant: float,
    de_constant: float,
    num_iter: int,
    scaling_factor_arrow: float = 1,
    head_width: float = 0.009,
    head_length: float = 0.009,
    lw: float = 1,
    matrix_size: int = 100,
    comet_name: str = "comet",
) -> None:
    """
    Plot isofote for the comet and Monte Carlo simulation data.

    Parameters
    ----------
    x1 : float
        Starting x-coordinate for the plot.
    x2 : float
        Ending x-coordinate for the plot.
    y1 : float
        Starting y-coordinate for the plot.
    y2 : float
        Ending y-coordinate for the plot.
    comet : np.ndarray
        2D array representing the comet image data.
    montecarlo_simulation : np.ndarray
        2D array of Monte Carlo simulation data.
    max_montecarlo_value : float
        Maximum value in the Monte Carlo simulation data.
    plot_multiplier : float
        Multiplier for scaling the plot coordinates.
    om : float
        X-coordinate for the arrow origin.
    am : float
        Y-coordinate for the arrow origin.
    xi : float
        Scaling factor for the arrow length.
    multiplier : float
        Multiplier for scaling the simulation data.
    levels : list
        Contour levels for the plot.
    ra_constant : float
        Right ascension constant in degrees.
    de_constant : float
        Declination constant in degrees.
    num_iter : int
        Number of iterations for the simulation.
    scaling_factor_arrow : float, optional
        Scaling factor for the arrow length (default is 1).
    head_width : float, optional
        Width of the arrow head (default is 0.009).
    head_length : float, optional
        Length of the arrow head (default is 0.009).
    lw : float, optional
        Line width for the arrows (default is 1).
    matrix_size : int, optional
        Size of the plot grid (default is 100).
    comet_name : str, optional
        Name of the comet for file naming (default is 'comet').

    Returns
    -------
    None
    """
    logger.info("Plotting...")

    # Set the output of the plot to PNG file
    plt.figure()
    plt.rcParams["savefig.format"] = "png"

    # Graph layout configuration
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)  # Set the size of the graph in inches
    plt.subplots_adjust(left=0.2, bottom=0.09, right=0.9, top=0.99)

    # Define the data to be plotted
    x = plot_multiplier * (x1 + (x2 - x1) * np.arange(matrix_size) / matrix_size)
    y = plot_multiplier * (y1 + (y2 - y1) * np.arange(matrix_size) / matrix_size)

    # Convert to double type for parameters om, am, xi
    om = np.double(om)
    am = np.double(am)
    xi = np.double(xi)

    # First contour plot for the comet
    _ = ax.contour(x, y, comet, levels=levels, colors="gray", linewidths=1)
    ax.set_xlabel("X-axis label")
    ax.set_ylabel("To North Pole (10⁶ km)")
    ax.tick_params(axis="x", width=4)
    ax.tick_params(axis="y", width=4)

    # Second contour plot for the Monte Carlo simulation
    _ = ax.contour(
        x,
        y,
        multiplier * montecarlo_simulation / max_montecarlo_value,
        levels=levels,
        colors="k",
        linewidths=1,
    )

    # Convert right ascension and declination constants to radians
    ra = (np.pi * ra_constant) / 180
    de = (np.pi * de_constant) / 180

    # Save the current axis limits
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    # X axis
    dn = 3
    if om < x_limits[0] or om > x_limits[1]:
        om = x_limits[0] + (x_limits[1] / dn)
    if am < x_limits[0] or am > x_limits[1]:
        am = x_limits[0] + (x_limits[1] / dn)
    # Y axis
    if om < y_limits[0] or om > y_limits[1]:
        om = y_limits[0] + (y_limits[1] / dn)
    if am < y_limits[0] or am > y_limits[1]:
        am = y_limits[0] + (y_limits[1] / dn)

    dx1 = -xi * np.sin(ra) * scaling_factor_arrow
    dy1 = xi * np.cos(ra) * scaling_factor_arrow

    dx2 = -xi * np.sin(de) * scaling_factor_arrow
    dy2 = xi * np.cos(de) * scaling_factor_arrow

    dn2 = 10
    max_x = x_limits[1] - x_limits[0]
    max_y = y_limits[1] - y_limits[0]

    # Check both positive and negative values
    if abs(dx1) > max_x:
        dx1 = np.sign(dx1) * (max_x / dn2)
    if abs(dy1) > max_y:
        dy1 = np.sign(dy1) * (max_y / dn2)

    if abs(dx2) > max_x:
        dx2 = np.sign(dx2) * (max_x / dn2)
    if abs(dy2) > max_y:
        dy2 = np.sign(dy2) * (max_y / dn2)

    ax.arrow(om, am, dx1, dy1, head_width=head_width, head_length=head_length, fc="k", ec="k", lw=lw)
    ax.arrow(om, am, dx2, dy2, head_width=head_width, head_length=head_length, fc="grey", ec="grey", lw=lw)

    # Check if the output directory exists, otherwise create it
    output_dir = f"output/{comet_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot as a PNG file
    plt.savefig(
        f"{output_dir}/{comet_name}_{int(num_iter / 1000)}K_{int(multiplier / 1000)}K_new.png", dpi=1200
    )
    plt.close()  # Close the figure to free memory
    logger.info("Plot saved successfully.")


def check_comet_portion_size(
    comet_image: pd.DataFrame, portion_comet: tuple[int, int, int, int], minimum_size: int = 100
) -> tuple[int, int, int, int]:
    """
    Check and adjust the portion size of the comet image to ensure it meets the minimum size requirement.

    Parameters
    ----------
    comet_image : pd.DataFrame
        DataFrame containing the comet image data.
    portion_comet : tuple[int, int, int, int]
        The portion defined as (start_row, end_row, start_col, end_col).
    minimum_size : int, optional
        The minimum required size for the portion (default is 100).

    Returns
    -------
    tuple[int, int, int, int]
        Updated portion coordinates ensuring the minimum size.
    """
    try:
        current_height = portion_comet[1] - portion_comet[0]
        current_width = portion_comet[3] - portion_comet[2]

        # Check and adjust height if necessary
        if current_height + 1 < minimum_size:
            logger.warning(f"Initial portion: {portion_comet}")
            logger.warning(f"Height is too small ({current_height} < {minimum_size}). Recomputing...")
            portion_comet = recompute_portion(comet_image, portion_comet, minimum_size, 0, 1, 0)

        # Check and adjust width if necessary
        if current_width + 1 < minimum_size:
            logger.warning(f"Initial portion: {portion_comet}")
            logger.warning(f"Width is too small ({current_width} < {minimum_size}). Recomputing...")
            portion_comet = recompute_portion(comet_image, portion_comet, minimum_size, 2, 3, 1)

        # logger.info(f"Final portion after checks: {portion_comet}")

    except Exception as _:
        # logger.error(f"Error while checking comet portion size: {e}")
        raise

    return portion_comet


def recompute_portion(
    comet_image: pd.DataFrame,
    portion_comet: tuple[int, int, int, int],
    minimum_size: int,
    start_idx: int,
    end_idx: int,
    axis: int,
) -> tuple[int, int, int, int]:
    """
    Recompute the portion size of the comet image to ensure it meets the minimum size requirement.

    Parameters
    ----------
    comet_image : pd.DataFrame
        DataFrame containing the comet image data.
    portion_comet : tuple[int, int, int, int]
        The portion defined as (start_row, end_row, start_col, end_col).
    minimum_size : int
        The minimum required size for the portion.
    start_idx : int
        The starting index of the portion to be adjusted.
    end_idx : int
        The ending index of the portion to be adjusted.
    axis : int
        The axis along which the adjustment is to be made (0 for rows, 1 for columns).

    Returns
    -------
    tuple[int, int, int, int]
        Updated portion coordinates ensuring the minimum size.
    """
    try:
        # Convert to a mutable list
        portion_comet = list(portion_comet)

        # Compute the current size and required adjustment
        current_size = portion_comet[end_idx] - portion_comet[start_idx]
        extra_size = minimum_size - current_size

        # Adjust height while staying within image boundaries
        if comet_image.shape[axis] < portion_comet[end_idx] + extra_size:
            logger.warning("Error out of range.")

        # Adjust the starting and ending indices while respecting image boundaries
        portion_comet[end_idx] = min(comet_image.shape[axis], portion_comet[end_idx] + extra_size)

        logger.info(f"Recomputed portion: {tuple(portion_comet)}")
        return tuple(portion_comet)
    except Exception as _:
        # logger.error(f"Error while recomputing portion: {e}")
        raise


def clip_arrow(
    om: float, am: float, dx: float, dy: float, x_limits: tuple[float, float], y_limits: tuple[float, float]
) -> tuple[float, float]:
    """
    Clip the arrow coordinates to ensure they stay within the plot limits.

    Parameters
    ----------
    om : float
        X-coordinate for the arrow origin.
    am : float
        Y-coordinate for the arrow origin.
    dx : float
        Change in x-coordinate for the arrow.
    dy : float
        Change in y-coordinate for the arrow.
    x_limits : tuple[float, float]
        Tuple containing the minimum and maximum x-coordinate limits.
    y_limits : tuple[float, float]
        Tuple containing the minimum and maximum y-coordinate limits.

    Returns
    -------
    tuple[float, float]
        The clipped change in x and y coordinates for the arrow.
    """
    # Calculate the final coordinates of the arrow
    x_end = om + dx
    y_end = am + dy

    # Clip the final coordinates to the plot limits
    if x_end < x_limits[0]:
        x_end = x_limits[0]
    elif x_end > x_limits[1]:
        x_end = x_limits[1]

    if y_end < y_limits[0]:
        y_end = y_limits[0]
    elif y_end > y_limits[1]:
        y_end = y_limits[1]

    # Recalculate the length of the arrow after clipping
    dx_clipped = x_end - om
    dy_clipped = y_end - am

    return dx_clipped, dy_clipped
