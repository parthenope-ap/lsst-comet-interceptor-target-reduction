import argparse
import time
from typing import Any

import numpy as np
from lsst_comet_interceptor_target_reduction.comet import Comet
from lsst_comet_interceptor_target_reduction.comet_bean_dao import CometBeanDAO
from lsst_comet_interceptor_target_reduction.logger import Logger
from lsst_comet_interceptor_target_reduction.montecarlo import MonteCarlo, scoring_fn
from lsst_comet_interceptor_target_reduction.plotter import plot_isofote

from parse_comet import parse_comet

logger = Logger()


def main(args: dict[str, Any]) -> None:
    """
    Main function to perform grid search for comet parameters.

    Parameters
    ----------
    args : dict[str, Any]
        Dictionary of command line arguments.
    """

    # Initialize the parameters
    params, comet_image, constants = parse_comet(args)
    comet_name = params["comet_name"]
    num_iter = params["mc_kwargs"]["num_iter"]

    plot_isofote_kwargs = params["plot_isofote_kwargs"]
    mc_kwargs = params["mc_kwargs"]
    scoring_fn_kwargs = params["scoring_fn_kwargs"]

    start_time = time.time()
    dao = CometBeanDAO(comet_name=params["comet_name"])
    comet_bean = dao.create_comet_bean(constants)
    comet = Comet(comet_bean)

    # Create a Monte Carlo object with the comet
    monte_carlo = MonteCarlo(comet=comet, **mc_kwargs)

    # Fit the model
    f = monte_carlo.fit(verbose=True)

    # compute fit error
    ra, g, f, result, de, fit_error = scoring_fn(
        comet_image=comet_image,
        montecarlo_simulation=f,
        perihelion_distance=comet.get_perihelion_distance(),
        eccentricity=comet.get_eccentricity(),
        anomaly=comet.get_anomaly(),
        **scoring_fn_kwargs,
    )
    logger.info("****** Some result *******")
    logger.info(f"RA: {ra}")
    logger.info(f"Result: {result}")
    logger.info(f"De: {de}")
    logger.info(f"Fit error: {fit_error}")
    logger.info("**************************")

    g = np.transpose(g)
    f = np.transpose(f)

    plot_isofote(
        x1=monte_carlo.x1,
        x2=monte_carlo.x2,
        y1=monte_carlo.y1,
        y2=monte_carlo.y2,
        comet=g,
        montecarlo_simulation=f,
        max_montecarlo_value=ra,
        **plot_isofote_kwargs,
    )

    # Compute the execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # output destination file
    file_path = f"output/{comet_name}/test_time_results.txt"

    # Open the file in append mode and write the results
    with open(file_path, "a") as file:
        file.write(f"{num_iter} iteration, time: {round(execution_time / 60):.2f} min\n")

    logger.info("All operations completed successfully.")
    logger.info(f"Execution time: {execution_time:.2f} second")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--comet_name", required=True, help="Name of the comet")
    args = vars(ap.parse_args())
    main(args)

# python main.py --comet_name C2016Q2
