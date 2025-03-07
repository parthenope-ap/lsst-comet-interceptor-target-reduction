import argparse
import concurrent
import concurrent.futures
import gc
import time
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
from lsst_comet_interceptor_target_reduction.comet import Comet
from lsst_comet_interceptor_target_reduction.comet_bean_dao import CometBeanDAO
from lsst_comet_interceptor_target_reduction.logger import Logger
from lsst_comet_interceptor_target_reduction.montecarlo import MonteCarlo, scoring_fn
from lsst_comet_interceptor_target_reduction.plotter import plot_isofote
from tqdm import tqdm

from parse_comet import parse_comet

logger = Logger()


def perform_mc_search(
    comet: Comet, comet_image: pd.DataFrame, param_3: float, exp_val: float, k: float, params: dict[str, Any]
) -> tuple[MonteCarlo, float, np.ndarray, np.ndarray, Any, float, float, float, float]:
    """
    Perform a Monte Carlo search for the given comet parameters.

    Parameters
    ----------
    comet : Comet
        The comet object.
    comet_image : pd.DataFrame
        Dataframe containing the comet image data.
    param_3 : float
        The third parameter for the Monte Carlo simulation.
    exp_val : float
        The exponent value for the Monte Carlo simulation.
    k : float
        The k parameter for the Monte Carlo simulation.
    params : dict[str, Any]
        Dictionary containing various parameters for the simulation.

    Returns
    -------
    tuple[MonteCarlo, float, np.ndarray, np.ndarray, Any, float, float, float, float]
        A tuple containing the Monte Carlo object, RA value, g array, f array, result,
        fit error, param_3, exp_val, and k.
    """

    mc_kwargs = params["mc_kwargs"]
    scoring_fn_kwargs = params["scoring_fn_kwargs"]

    # Create a Monte Carlo object with the comet
    monte_carlo = MonteCarlo(comet=comet, param_3=param_3, exp_val=exp_val, k=k, **mc_kwargs)

    # Compute the model
    f = monte_carlo.fit()

    # Compute fit error
    ra, g, f, result, de, fit_error = scoring_fn(
        comet_image=comet_image,
        montecarlo_simulation=f,
        perihelion_distance=comet.get_perihelion_distance(),
        eccentricity=comet.get_eccentricity(),
        anomaly=comet.get_anomaly(),
        **scoring_fn_kwargs,
    )

    return monte_carlo, ra, g, f, result, fit_error, param_3, exp_val, k


def main(args):
    """
    Main function to perform grid search for comet parameters.

    Parameters
    ----------
    args : dict[str, Any]
        Dictionary of command line arguments.
    """

    # Initialize the parameters
    pprint(args)
    params, comet_image = parse_comet(args)
    comet_name = params["comet_name"]
    num_iter = params["mc_kwargs"]["num_iter"]

    grids = {
        "param_3": np.linspace(
            args["start_param_3"],
            args["stop_param_3"],
            args["num_samples_param_3"] + 1
            if args["num_samples_param_3"] > 1
            else args["num_samples_param_3"],
        ),
        "exp_val": np.linspace(
            args["start_exp_val"],
            args["stop_exp_val"],
            args["num_samples_exp_val"] + 1
            if args["num_samples_exp_val"] > 1
            else args["num_samples_exp_val"],
        ),
        "k": np.linspace(
            args["start_k"],
            args["stop_k"],
            args["num_samples_k"] + 1 if args["num_samples_k"] > 1 else args["num_samples_k"],
        ),
    }

    plot_isofote_kwargs = params["plot_isofote_kwargs"]

    start_time = time.time()
    dao = CometBeanDAO(comet_name=params["comet_name"])
    comet_bean = dao.create_comet_bean()
    comet = Comet(comet_bean)

    best_fit = {
        "monte_carlo": None,
        "ra": None,
        "g": None,
        "f": None,
        "fit_error": np.inf,
        "K": None,
        "EXP_VAL": None,
        "MC_PARAM_3": None,
    }

    logger.info("Starting grid search...")
    logger.info("GRID PARAMETERS:")
    logger.info(
        f"param_3: np.linspace({grids['param_3'][0]}, {grids['param_3'][-1]}, {len(grids['param_3'])})"
    )
    logger.info(
        f"exp_val: np.linspace({grids['exp_val'][0]}, {grids['exp_val'][-1]}, {len(grids['exp_val'])})"
    )
    logger.info(f"k: np.linspace({grids['k'][0]}, {grids['k'][-1]}, {len(grids['k'])})")
    total_iterations = len(grids["param_3"]) * len(grids["exp_val"]) * len(grids["k"])
    logger.info(f"Total number of iterations: {total_iterations}")

    pbar = tqdm(
        total=len(grids["param_3"]) * len(grids["exp_val"]) * len(grids["k"]), desc="Grid search progress"
    )
    perform_mc_search_kwargs = [
        (param_3, exp_val, k)
        for param_3 in grids["param_3"]
        for exp_val in grids["exp_val"]
        for k in grids["k"]
    ]

    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(perform_mc_search, comet, comet_image, *mc_search_kwargs, params): mc_search_kwargs
            for mc_search_kwargs in perform_mc_search_kwargs
        }

        for future in concurrent.futures.as_completed(futures):
            monte_carlo, ra, g, f, result, fit_error, param_3, exp_val, k = future.result()
            pbar.update(1)
            if fit_error < best_fit["fit_error"]:
                best_fit["monte_carlo"] = monte_carlo
                best_fit["ra"] = ra
                best_fit["g"] = g
                best_fit["f"] = f
                best_fit["result"] = result
                best_fit["fit_error"] = fit_error
                best_fit["MC_PARAM_3"] = param_3
                best_fit["EXP_VAL"] = exp_val
                best_fit["K"] = k
            del monte_carlo, ra, g, f, result, fit_error, param_3, exp_val, k
            del futures[future]
            gc.collect()

    pbar.close()
    logger.info("Grid search completed.")

    # dump best fit
    logger.info("Best fit:")
    logger.info(f"Fit error: {best_fit['fit_error']}")
    logger.info(f"EXP_VAL: {best_fit['EXP_VAL']}")
    logger.info(f"K: {best_fit['K']}")
    logger.info(f"MC_PARAM_3: {best_fit['MC_PARAM_3']}")
    logger.info(f"RA: {best_fit['ra']}")
    logger.info(f"Result: {best_fit['result']}")

    # get the MonteCarlo object corresponding to the best fit
    monte_carlo = best_fit["monte_carlo"]
    g = np.transpose(best_fit["g"])
    f = np.transpose(best_fit["f"])
    ra = best_fit["ra"]

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
    ap.add_argument("-c", "--comet_name", required=True, help="Comet name")

    # grid search arguments
    ap.add_argument(
        "--start_param_3",
        required=True,
        type=float,
        default=0.0,
        help="Starting element of the grid search for param_3",
    )
    ap.add_argument(
        "--stop_param_3",
        required=True,
        type=float,
        default=0.05,
        help="Stopping element of the grid search for param_3",
    )
    ap.add_argument(
        "--num_samples_param_3",
        required=True,
        type=int,
        default=50,
        help="Number of grid samples for param_3",
    )
    ap.add_argument(
        "--start_exp_val",
        required=True,
        type=float,
        default=-5.0,
        help="Starting element of the grid search for exp_val",
    )
    ap.add_argument(
        "--stop_exp_val",
        required=True,
        type=float,
        default=0.0,
        help="Stopping element of the grid search for exp_val",
    )
    ap.add_argument(
        "--num_samples_exp_val",
        required=True,
        type=int,
        default=50,
        help="Number of grid samples for exp_val",
    )
    ap.add_argument(
        "--start_k", required=True, type=float, default=1.0, help="Starting element of the grid search for k"
    )
    ap.add_argument(
        "--stop_k", required=True, type=float, default=2.0, help="Stopping element of the grid search for k"
    )
    ap.add_argument(
        "--num_samples_k", required=True, type=int, default=10, help="Number of grid samples for k"
    )
    args = vars(ap.parse_args())
    main(args)

# Single-configuration test
# python parallelized_main.py --comet_name C2016Q2 \
# --start_param_3 0.01 --stop_param_3 0.01 --num_samples_param_3 1 \
# --start_exp_val -1.0 --stop_exp_val -1.0 --num_samples_exp_val 1 \
# --start_k 1.0 --stop_k 1.0 --num_samples_k 1

# Grid-search test
# python parallelized_main.py --comet_name C2016Q2 \
# --start_param_3 0.0 --stop_param_3 0.05 --num_samples_param_3 50 \
# --start_exp_val 0.0 --stop_exp_val 5.0 --num_samples_exp_val 50 \
# --start_k 1.0 --stop_k 2.0 --num_samples_k 10
