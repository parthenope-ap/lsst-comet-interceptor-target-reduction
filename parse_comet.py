import importlib
from typing import Any

import pandas as pd
from lsst_comet_interceptor_target_reduction.comet_bean_dao import CONSTANTS
from lsst_comet_interceptor_target_reduction.logger import Logger

logger = Logger()


def parse_comet(args: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Initialize parameters for the comet analysis.

    Parameters
    ----------
    args : dict[str, Any]
        Dictionary of arguments passed to the script.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        Dictionary of parameters for the comet analysis.
        DataFrame containing the comet image data.
    """
    comet_name = args["comet_name"]
    constants_module = importlib.import_module(f"input.{comet_name}.{comet_name}")

    params = {
        "comet_name": comet_name,
        "mc_kwargs": {
            "dim_matrix_image": getattr(constants_module, "nc", 100),
            "depend": getattr(constants_module, "DEPEND_MONTECARLO", False),
            "num_iter": getattr(constants_module, "NUM_ITER", 10000),
            "t_size": getattr(constants_module, "T_SIZE", None),
            "param_00": getattr(constants_module, "MC_PARAM_00", None),
            "param_01": getattr(constants_module, "MC_PARAM_01", None),
            "param_02": getattr(constants_module, "MC_PARAM_02", None),
            "param_03": getattr(constants_module, "MC_PARAM_03", None),
            # second free parameter: dust velocity v_d (km/s, 1AU)
            "param_3": getattr(constants_module, "MC_PARAM_3", None),
            # first free parameter: -1/2 sigma ** 2
            "exp_val": getattr(constants_module, "EXP_VAL", None),
            # third free parameter k (heliocentric distance exponent)
            "k": getattr(constants_module, "K", None),
        },
        "scoring_fn_kwargs": {
            "matrix_size": getattr(constants_module, "nc", 100),
            "portion_comet": getattr(constants_module, "PORTION_COMET", None),
            "minimum_size": getattr(constants_module, "MININUM_SIZE", None),
            "binning": getattr(constants_module, "BINNING", None),
            "sky_background_value": getattr(constants_module, "SKY_BACKGROUND_VALUE", None),
            "multiplier": getattr(constants_module, "MULTIPLIER", None),
            "threshold_min": getattr(constants_module, "THRESHOLD_MIN", None),
        },
        "plot_isofote_kwargs": {
            "ra_constant": getattr(constants_module, "RA_CONSTANT", None),
            "de_constant": getattr(constants_module, "DE_CONSTANT", None),
            "om": getattr(constants_module, "OM_RECYCLED", None),
            "am": getattr(constants_module, "AM_RECYCLED", None),
            "xi": getattr(constants_module, "XI_RECYCLED", None),
            "levels": getattr(constants_module, "LEVELS", None),
            "scaling_factor_arrow": getattr(constants_module, "SCALING_FACTOR_ARROW", None),
            "head_width": getattr(constants_module, "HEAD_WIDTH", None),
            "head_length": getattr(constants_module, "HEAD_LENGTH", None),
            "lw": getattr(constants_module, "LW", None),
            "plot_multiplier": getattr(constants_module, "PLOT_MULTIPLICATOR", None),
            "num_iter": getattr(constants_module, "NUM_ITER", None),
            "comet_name": comet_name,
            "multiplier": getattr(constants_module, "MULTIPLIER", None),
            "matrix_size": getattr(constants_module, "nc", None),
        },
    }

    try:
        comet_image = pd.read_csv(f"input/{comet_name}/{comet_name}.txt", sep="\\s+")
    except FileNotFoundError as e:
        logger.error(f"File {comet_name}/{comet_name}.txt not found.")
        raise e

    loaded_constants = {const: getattr(constants_module, const, None) for const in CONSTANTS}

    return params, comet_image, loaded_constants
