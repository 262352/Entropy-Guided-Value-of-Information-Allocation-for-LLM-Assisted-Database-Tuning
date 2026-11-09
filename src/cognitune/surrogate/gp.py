from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from cognitune.search_space.space import SearchSpace


@dataclass
class PosteriorPrediction:
    mean: np.ndarray
    std: np.ndarray


class GPPerformanceSurrogate:
    def __init__(self, space: SearchSpace):
        self.space = space
        self.model: SingleTaskGP | None = None
        self.best_f: float | None = None

    def fit(self, configs: list[dict], utilities: list[float]) -> None:
        if len(configs) != len(utilities) or len(configs) < 2:
            raise ValueError("GP fitting requires at least two paired observations")
        train_x = torch.as_tensor(self.space.encode_many(configs), dtype=torch.double)
        train_y = torch.as_tensor(utilities, dtype=torch.double).unsqueeze(-1)
        torch.manual_seed(0)
        covariance = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]))
        self.model = SingleTaskGP(train_x, train_y, covar_module=covariance, outcome_transform=Standardize(m=1))
        likelihood = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(
            likelihood,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"step_limit": 75},
            max_attempts=1,
        )
        self.model.eval()
        self.best_f = float(max(utilities))

    def posterior(self, configs: list[dict]) -> PosteriorPrediction:
        model = self._require_model()
        x = torch.as_tensor(self.space.encode_many(configs), dtype=torch.double)
        with torch.no_grad():
            posterior = model.posterior(x)
        return PosteriorPrediction(
            posterior.mean.squeeze(-1).cpu().numpy(),
            posterior.variance.clamp_min(0).sqrt().squeeze(-1).cpu().numpy(),
        )

    def log_expected_improvement(self, configs: list[dict]) -> np.ndarray:
        model = self._require_model()
        if not configs:
            return np.empty(0)
        x = torch.as_tensor(self.space.encode_many(configs), dtype=torch.double).unsqueeze(-2)
        acquisition = LogExpectedImprovement(model=model, best_f=self.best_f)
        with torch.no_grad():
            return acquisition(x).cpu().numpy().reshape(-1)

    def log_ei(self, configs: list[dict]) -> np.ndarray:
        return self.log_expected_improvement(configs)

    def joint_posterior(self, configs: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
        """Latent posterior in utility units and observation-noise variance.

        Keep correlations: independent marginal draws give incorrect decisions
        about both conditional improvement and the best proposal source.
        """
        model = self._require_model()
        x = torch.as_tensor(self.space.encode_many(configs), dtype=torch.double)
        with torch.no_grad():
            latent = model.posterior(x)
            observed = model.posterior(x, observation_noise=True)
            mean = latent.mean.squeeze(-1).cpu().numpy()
            covariance = latent.mvn.covariance_matrix.cpu().numpy()
            noise = (observed.variance - latent.variance).mean().clamp_min(1e-12).item()
        return mean, covariance, float(noise)

    def log_noisy_expected_improvement(self, configs: list[dict], samples: int = 128, seed: int = 0) -> np.ndarray:
        """q=1 qLogNEI, integrating uncertainty at every training baseline."""
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.sampling.normal import IIDNormalSampler

        if not configs:
            return np.empty(0)
        model = self._require_model()
        acquisition = qLogNoisyExpectedImprovement(
            model, X_baseline=model.train_inputs[0],
            sampler=IIDNormalSampler(torch.Size([samples]), seed=seed),
            prune_baseline=True,
        )
        values = []
        # Bounded batch sizes keep 60-dimensional real runs within CPU memory.
        with torch.no_grad():
            for start in range(0, len(configs), 64):
                x = torch.as_tensor(self.space.encode_many(configs[start:start + 64]), dtype=torch.double)
                values.extend(acquisition(x.unsqueeze(-2)).cpu().numpy().reshape(-1))
        return np.asarray(values)

    def predict(self, configs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.posterior(configs)
        return prediction.mean, prediction.std

    def condition_improvement(self, incumbent: dict, candidates_by_knob: dict[str, list[dict]], beta: float = 1.0) -> dict[str, float]:
        base_prediction = self.posterior([incumbent])
        base = float(base_prediction.mean[0])
        result = {}
        for name, configs in candidates_by_knob.items():
            if not configs:
                result[name] = 0.0
                continue
            prediction = self.posterior(configs)
            result[name] = max(0.0, float(np.max(prediction.mean - base + beta * prediction.std)))
        return result

    def _require_model(self) -> SingleTaskGP:
        if self.model is None:
            raise RuntimeError("surrogate has not been fitted")
        return self.model
