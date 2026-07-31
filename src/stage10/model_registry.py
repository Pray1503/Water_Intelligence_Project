"""Stage 10 - Layer 3: Configuration Driven Model Registry.

Instantiates ONLY the models marked enabled in configuration. Supports
RandomForestRegressor, XGBRegressor, and LightGBMRegressor. No training
happens in this layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.base import BaseEstimator

from src.stage10.exceptions import ModelRegistryError, UnknownModelError
from src.stage10.logging_utils import log_call

logger = logging.getLogger("stage10")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single candidate model.

    Attributes
    ----------
    name:
        Registered model name, e.g. "RandomForestRegressor".
    enabled:
        Whether this model should be instantiated.
    hyperparameters:
        Keyword arguments passed to the model constructor.
    random_state:
        Random seed. Applied only if provided.
    """

    name: str
    enabled: bool
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_state: int | None = None


@dataclass(frozen=True)
class ModelRegistry:
    """Container wrapping instantiated unfitted candidate models.

    Attributes
    ----------
    models:
        Mapping of model name -> unfitted scikit-learn compatible estimator.
    """

    models: dict[str, BaseEstimator]

    @property
    def enabled_model_names(self) -> list[str]:
        """Return list of instantiated model names."""
        return list(self.models.keys())

    @property
    def model_count(self) -> int:
        """Return the total number of instantiated models."""
        return len(self.models)

    def __getitem__(self, name: str) -> BaseEstimator:
        """Access instantiated model by name."""
        if name not in self.models:
            raise ModelRegistryError(
                f"Model '{name}' not found in registry. "
                f"Available models: {self.enabled_model_names}"
            )
        return self.models[name]


def _build_random_forest_regressor(**kwargs: Any) -> BaseEstimator:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**kwargs)


def _build_xgb_regressor(**kwargs: Any) -> BaseEstimator:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # noqa: BLE001
        raise ModelRegistryError(
            "XGBRegressor is enabled but the 'xgboost' package is not installed."
        ) from exc
    return XGBRegressor(**kwargs)


def _build_lightgbm_regressor(**kwargs: Any) -> BaseEstimator:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # noqa: BLE001
        raise ModelRegistryError(
            "LightGBMRegressor is enabled but the 'lightgbm' package is not installed."
        ) from exc
    return LGBMRegressor(**kwargs)


# Maps a registered model name to a builder function. Builder functions
# accept arbitrary keyword args and return an unfitted model instance.
MODEL_BUILDER_REGISTRY: dict[str, Callable[..., BaseEstimator]] = {
    "RandomForestRegressor": _build_random_forest_regressor,
    "XGBRegressor": _build_xgb_regressor,
    "LightGBMRegressor": _build_lightgbm_regressor,
}


@log_call
def validate_model_config(config: ModelConfig) -> None:
    """Raise UnknownModelError if config.name is not registered.

    Raises
    ------
    UnknownModelError
    """
    if config.name not in MODEL_BUILDER_REGISTRY:
        raise UnknownModelError(
            f"Unknown model '{config.name}'. Registered models: "
            f"{sorted(MODEL_BUILDER_REGISTRY)}"
        )


@log_call
def instantiate_model(config: ModelConfig) -> BaseEstimator:
    """Instantiate a single (unfitted) model from *config*.

    Parameters
    ----------
    config:
        Model configuration. Must have enabled=True to be meaningfully
        used by build_model_registry, but this function will instantiate
        regardless of the enabled flag if called directly.

    Returns
    -------
    An unfitted scikit-learn compatible estimator instance.

    Raises
    ------
    UnknownModelError
        If config.name is not a registered model.
    ModelRegistryError
        If the required third-party package is not installed, or
        instantiation otherwise fails.
    """
    validate_model_config(config)

    builder = MODEL_BUILDER_REGISTRY[config.name]
    kwargs = dict(config.hyperparameters)

    if config.random_state is not None:
        kwargs.setdefault("random_state", config.random_state)

    try:
        model = builder(**kwargs)
    except ModelRegistryError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelRegistryError(
            f"Failed to instantiate model '{config.name}' with "
            f"hyperparameters {kwargs}: {exc}"
        ) from exc

    logger.info(
        "Instantiated model '%s' (%d hyperparameters, random_state=%s)",
        config.name,
        len(config.hyperparameters),
        config.random_state,
    )
    return model


@log_call
def build_model_registry(configs: list[ModelConfig]) -> ModelRegistry:
    """Instantiate every ENABLED model in *configs*.

    Parameters
    ----------
    configs:
        List of ModelConfig entries, typically parsed from
        config.stage10.models.

    Returns
    -------
    ModelRegistry wrapping a dict mapping model name -> unfitted model instance,
    containing only enabled models.

    Raises
    ------
    UnknownModelError
    ModelRegistryError
        If no enabled models exist in the configuration.
    """
    registry_dict: dict[str, BaseEstimator] = {}

    for config in configs:
        if not config.enabled:
            logger.info("Skipping disabled model '%s'", config.name)
            continue
        registry_dict[config.name] = instantiate_model(config)

    if not registry_dict:
        raise ModelRegistryError(
            "Cannot build model registry: 0 models are enabled in the configuration."
        )

    logger.info(
        "Model registry successfully built with %d model(s)", len(registry_dict)
    )
    return ModelRegistry(models=registry_dict)


@log_call
def load_model_configs_from_dict(
    raw_configs: list[dict[str, Any]],
) -> list[ModelConfig]:
    """Parse a list of raw config dicts (as loaded from YAML) into
    ModelConfig instances.

    Each dict is expected to have keys: name, enabled, hyperparameters
    (optional), random_state (optional).

    Raises
    ------
    ModelRegistryError
        If a required key ('name' or 'enabled') is missing from an entry.
    """
    parsed: list[ModelConfig] = []
    for index, raw in enumerate(raw_configs):
        if "name" not in raw:
            raise ModelRegistryError(
                f"Model config at index {index} is missing 'name'."
            )
        if "enabled" not in raw:
            raise ModelRegistryError(
                f"Model config '{raw.get('name', '?')}' at index {index} is missing 'enabled'."
            )

        parsed.append(
            ModelConfig(
                name=str(raw["name"]),
                enabled=bool(raw["enabled"]),
                hyperparameters=dict(raw.get("hyperparameters", {})),
                random_state=raw.get("random_state"),
            )
        )

    return parsed
