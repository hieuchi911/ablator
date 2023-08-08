from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    PlotAnalysis,
)
from ablator.analysis.results import Results


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


@pytest.mark.order(1)
def test_analysis(tmp_path: Path, ablator_results):
    results: Results = ablator_results
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.mock_param": "Some Parameter",
    }
    numerical_name_remap = {
        "train_config.optimizer_config.arguments.lr": "Learning Rate",
    }
    analysis = PlotAnalysis(
        results,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_loss": "min"},
    )
    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_loss": "Val. Loss",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_loss", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )

    assert all(
        tmp_path.joinpath("linearplot", "val_loss", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )
    pass


# https://github.com/fostiropoulos/ablator/pull/35
def test_get_best_results_by_metric():

    data = {'path': ['path1', 'path1', 'path2', 'path2', 'path3'],
            'metric1': [0.1, 0.2, 0.3, None, 0.5],
            'metric2': [None, 0.6, 0.7, 0.8, 0.9]}
    input = pd.DataFrame(data)

    result = pd.DataFrame({
        'path': ['path1', 'path2', 'path3', 'path1', 'path2', 'path3'],
        'metric1': [0.1, 0.3, 0.5, 0.2, None, 0.5],
        'metric2': [None, 0.7, 0.9, 0.6, 0.8, 0.9],
        'best': ['metric1', 'metric1', 'metric1', 'metric2', 'metric2', 'metric2']
    })

    metricMap = {'metric1': Optim.min, 'metric2': Optim.max}
    # call the function to be tested
    actual_output = Analysis._get_best_results_by_metric(input, metricMap)

    assert actual_output.equals(result)


if __name__ == "__main__":
    import shutil
    from tests.ray_models.model import (
        WORKING_DIR,
        TestWrapper,
        MyErrorCustomModel,
        _make_config,
    )
    from tests.conftest import DockerRayCluster, _assert_error_msg
    from tests.mp.test_main import test_mp_run

    ray_cluster = DockerRayCluster()
    ray_cluster.setUp()
    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_mp_run = test_mp_run(
        tmp_path,
        _assert_error_msg,
        ray_cluster,
        TestWrapper(MyErrorCustomModel),
        _make_config,
        WORKING_DIR,
    )
    config = _make_config(tmp_path, search_space_limit=10)
    test_mp_run = Results(config, f"/tmp/save_dir/experiment_{config.uid}")
    test_analysis(tmp_path)
