#!/usr/bin/env python
r"""Script for fit model to model data.

Examples
--------
Example of call

.. code-block:: shell

    ./scripts/analysis-sin_sq_weinberg.py \
      --free-parameters survival_probability neutrino_per_fission_factor \
      --constrained-parameters survival_probability detector reactor background reactor_anue \
      --output fit-result.yaml
"""
import IPython
from argparse import ArgumentParser, Namespace
from pprint import pprint
from typing import TYPE_CHECKING

from models.experiment_v1 import model_experiment_v1
from dgm_fit.iminuit_minimizer import IMinuitMinimizer

from scripts import do_fit, filter_save_fit, update_dict_parameters

if TYPE_CHECKING:
    from dag_modelling.parameters import Parameter


def main(args: Namespace) -> None:

    # Initialize model
    model = model_experiment_v1(
        seed=args.seed,
        monte_carlo_mode=args.monte_carlo_mode,
        parameter_values=args.par,
    )

    # Initialize helpful variables and switch output of model
    # to Asimov (output 0) or Real data (output 1).
    storage = model.storage

    parameters_free = storage("parameters.free")
    parameters_constrained = storage("parameters.constrained")
    statistic = storage("outputs.statistic")

    # Choose statistic for minimization
    chi2 = statistic[f"{args.statistic}"]
    # Fill variable `minimization_parameters` free and constrained parameters,
    # if they are given
    minimization_parameters: dict[str, Parameter] = {}
    update_dict_parameters(minimization_parameters, args.free_parameters, parameters_free)
    if "covmat" not in args.statistic:
        update_dict_parameters(
            minimization_parameters,
            args.constrained_parameters,
            parameters_constrained,
        )
    elif args.constrained_parameters:
        raise Exception(f"Statistic {args.statistic} can not be used with constrained parameters")

    # Sometimes fit is unstable. And constraining of free parameters
    # might improve robustness of fit
    # TODO: remove parameters of interests (enues.sin_sq_weinberg) from minimization_parameters

    minimizer = IMinuitMinimizer(
        chi2, parameters=minimization_parameters, nbins=model.nbins, verbose=args.verbose > 1
    )

    print(f"Initial value of chi-squared: {chi2.data}")

    # TODO: add values for scanning
    # TODO: add for-cycle for setting enues.sin_sq_weinberg and do fit in each point (model.set_parameters)
    # TODO: add list for saving chi-squared value in each point of enues.sin_sq_weinberg
    # Start fitting
    result = do_fit(minimizer, model, args.n_iterations)

    if args.profile_parameters:
        errors_profiled = minimizer.profile_errors(args.profile_parameters)
        result["errorsdict_profiled"] = errors_profiled["errorsdict"]

    if args.output:
        filter_save_fit(result, args.output)

    pprint(result)

    if args.interactive:
        IPython.embed()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=0, action="count", help="verbosity level")

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--path-data",
        default=None,
        help="Path to data",
    )
    model.add_argument(
        "--par",
        nargs=2,
        action="append",
        default=[],
        help="set parameter value",
    )
    model.add_argument(
        "--monte-carlo-mode",
        "--mc",
        default="asimov",
        choices=["asimov", "normal-stats", "poisson"],
        help="Choose Monte-Carlo option",
    )
    model.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Choose seed for random generation, important in case of `monte_carlo_mode` != `asimov`",
    )
    model.add_argument(
        "--concatenation-mode",
        default="detector_period",
        choices=["detector", "detector_period"],
        help="Choose type of concatenation for final observation: by detector or by detector and period",
    )

    model.add_argument(
        "--interactive"
    )

    fit_options = parser.add_argument_group("fit", "Set fit procedure")
    fit_options.add_argument(
        "--data",
        default="asimov",
        choices=["asimov", "real"],
        help="choose data for fit",
    )
    fit_options.add_argument(
        "--constrain-osc-parameters",
        action="store_true",
        help="constrain oscillation parameters",
    )
    fit_options.add_argument(
        "--profile-parameters",
        action="extend",
        nargs="*",
        default=[],
        help="choose parameters for Minos profiling",
    )
    fit_options.add_argument(
        "--statistic",
        default="stat.chi2p",
        choices=[
            "stat.chi2p",
        ],
        help="choose chi-squared function for minimizer",
    )
    fit_options.add_argument(
        "--n-iterations",
        default=0,
        help="number of iterations of fit procedure, usefull only for iterative chi-squared",
    )
    fit_options.add_argument(
        "--free-parameters",
        default=[],
        nargs="*",
        help="add free parameters to minimization process",
    )
    fit_options.add_argument(
        "--constrained-parameters",
        default=[],
        nargs="*",
        help="add constrained parameters to minimization process",
    )

    outputs = parser.add_argument_group("outputs", "set outputs")
    outputs.add_argument(
        "--output",
        help="path to save full fit, yaml format",
    )

    args = parser.parse_args()

    main(args)
