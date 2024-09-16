from typing import Mapping

from dagflow.logger import logger

from .experiment_v0 import model_experiment_v0

_dayabay_models = {
    "v0": model_experiment_v0,
}


def available_models() -> tuple[str, ...]:
    return tuple(_dayabay_models.keys())


def load_model(version, model_options: Mapping | str = {}, **kwargs):
    if isinstance(model_options, str):
        from yaml import Loader, load

        model_options = load(model_options, Loader)

    if not isinstance(model_options, dict):
        raise RuntimeError(
            "model_options expects a python dictionary or yaml dictionary"
        )

    model_options = dict(model_options, **kwargs)

    logger.info(f"Execute Daya Bay model {version}")
    try:
        cls = _dayabay_models[version]
    except KeyError:
        raise RuntimeError(
            f"Invalid model version {version}. Available models: {', '.join(sorted(_dayabay_models.keys()))}"
        )

    return cls(**model_options)
