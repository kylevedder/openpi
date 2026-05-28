import dataclasses
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training import config as _config

from . import train


class _TinyModel(_model.BaseModel):
    def __init__(self, config: "_TinyModelConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.proj = nnx.Linear(config.action_dim, config.action_dim, rngs=rngs)

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> jax.Array:
        del rng, train
        pred = self.proj(observation.state)
        pred = jnp.broadcast_to(pred[:, None, :], actions.shape)
        return jnp.mean(jnp.square(pred - actions), axis=-1)

    def sample_actions(self, rng: jax.Array, observation: _model.Observation, **kwargs) -> _model.Actions:
        del rng, kwargs
        pred = self.proj(observation.state)
        return jnp.broadcast_to(pred[:, None, :], (pred.shape[0], self.action_horizon, self.action_dim))


@dataclasses.dataclass(frozen=True)
class _TinyModelConfig(_model.BaseModelConfig):
    action_dim: int = 2
    action_horizon: int = 2
    max_token_len: int = 4

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    def create(self, rng: jax.Array) -> _TinyModel:
        return _TinyModel(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": jax.ShapeDtypeStruct([batch_size, 4, 4, 3], jnp.float32),
                },
                image_masks={
                    "base_0_rgb": jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        model=_TinyModelConfig(),
        num_workers=0,
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        tracking_backend="none",
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
