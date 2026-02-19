from __future__ import annotations

from .base import Scenario
from .collinearity import CollinearityScenario
from .invariance_demo import InvarianceDemoScenario
from .linear_regression_heteroskedastic import LinearRegressionHeteroskedasticScenario
from .linear_regression import LinearRegressionScenario
from .logistic_clustered import LogisticClusteredScenario
from .logistic_local_alternatives import LogisticLocalAlternativesScenario
from .logistic_separation import LogisticSeparationScenario
from .logistic_small_sample import LogisticSmallSampleScenario
from .normal_mean_known_sigma import NormalMeanKnownSigmaScenario
from .phacking import PHackingScenario
from .poisson_misspec import PoissonMisspecScenario
from .wald_runtime_batch import WaldRuntimeBatchScenario
from .zip_boundary import ZIPBoundaryScenario


def get_all_scenarios() -> list[Scenario]:
    return [
        NormalMeanKnownSigmaScenario(),
        LinearRegressionScenario(tested_indices=[1], name_suffix="single constraint"),
        LinearRegressionScenario(tested_indices=[1, 2], name_suffix="multiple constraints"),
        LogisticSeparationScenario(),
        LogisticSmallSampleScenario(),
        WaldRuntimeBatchScenario(),
        ZIPBoundaryScenario(),
        CollinearityScenario(),
        PoissonMisspecScenario(),
        InvarianceDemoScenario(),
        PHackingScenario(),
        LinearRegressionHeteroskedasticScenario(),
        LogisticClusteredScenario(),
        LogisticLocalAlternativesScenario(),
    ]


__all__ = [
    "Scenario",
    "NormalMeanKnownSigmaScenario",
    "LinearRegressionScenario",
    "LogisticSeparationScenario",
    "LogisticSmallSampleScenario",
    "WaldRuntimeBatchScenario",
    "ZIPBoundaryScenario",
    "CollinearityScenario",
    "PoissonMisspecScenario",
    "InvarianceDemoScenario",
    "PHackingScenario",
    "LinearRegressionHeteroskedasticScenario",
    "LogisticClusteredScenario",
    "LogisticLocalAlternativesScenario",
    "get_all_scenarios",
]
