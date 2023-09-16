import logging
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ablator.analysis.plot import Plot
from ablator.config.proto import Optim

logger = logging.getLogger(__name__)


class Categorical(Plot):
    """
    This class is for preparing the results that are associated with each categorical attribute to be studied
    (e.g., grouping metric results with each of the attributes). Its constructor takes in as input positional
    arguments or keyword arguments from the base class `Plot`. Possible arguments are listed in the Parameters
    section. The Attributes section lists its own attributes as well as those that are inherited.

    Parameters
    ----------
    *args : ty.Any
        Positional arguments to pass to the base class ``Plot``. These arguments are:

        - ``metric`` : ``pd.Series`` -  The ablation study metric values to plot,

        - ``attributes`` : ``pd.Series`` -  The ablation study attributes values to plot,

        - ``metric_obj_fn`` : ``Optim`` -  The metric optimization direction,

        - ``y_axis`` : ``str, optional`` -  The y-axis label (metric name), by default ``None``,

        - ``x_axis`` : ``str, optional`` -  The x-axis label (attribute name), by default ``None``,

        - ``x_ticks`` : ``list[str], optional`` -  The x-axis ticks, by default ``None``,

        - ``ax`` : ``Axes, optional`` -  The axes to plot on, by default ``None``.
    **kwargs : ty.Any
        Keyword arguments to pass to the base class ``Plot``. These arguments are:

        - ``metric`` : ``pd.Series`` -  The ablation study metric values to plot,

        - ``attributes`` : ``pd.Series`` -  The ablation study attributes values to plot,

        - ``metric_obj_fn`` : ``Optim`` -  The metric optimization direction,

        - ``y_axis`` : ``str, optional`` -  The y-axis label (metric name), by default ``None``,

        - ``x_axis`` : ``str, optional`` -  The x-axis label (attribute name), by default ``None``,

        - ``x_ticks`` : ``list[str], optional`` -  The x-axis ticks, by default ``None``,

        - ``ax`` : ``Axes, optional`` -  The axes to plot on, by default ``None``.

    Attributes
    ----------
    metric : pd.Series
        The ablation study metric values to plot (with null value removed).
    attributes : pd.Series
        The ablation study attributes values to plot (with null metric value removed).
    metric_obj_fn : Optim
        The metric optimization direction.
    y_axis : str
        The y-axis label (metric name).
    x_axis : str
        The x-axis label (attribute name).
    x_ticks : list[str]
        The x-axis ticks.
    figure : Figure
        The figure to plot on. If `None`, a new figure of size ``(4,4)`` will be created.
    ax : Axes
        The axes to plot on. If `None`, a new axis will be created as the first subplot
        in the first cell and first column of a `1x1` grid.
    DATA_TYPE : str
        The attribute data type. In this case, it is ``"categorical"``.
    attribute_metric_map : dict[str, pd.Series]
        A dictionary mapping attribute values to metric values.

    """

    DATA_TYPE: str = "categorical"

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        super().__init__(*args, **kwargs)
        self.attribute_metric_map = self._make_attribute_metric_map(
            self.metric, self.attributes
        )

    @classmethod
    def _make_attribute_metric_map(
        cls,
        metric: pd.Series,
        attributes: pd.Series,
    ) -> dict[str, pd.Series]:
        unique_values = attributes.unique()
        metrics: dict[str, pd.Series] = {}

        if None in unique_values:
            unique_values = list(filter(None, unique_values))
            none_name = "None"
            if "None" in unique_values:
                logger.warning(
                    "`None` name is present as categorical value as well as np.nan."
                )
                none_name = "Type(None)"
                assert none_name not in unique_values, (
                    f"{none_name} is also present as a categorical. Highly "
                    "unlikely it is by accident."
                )
            metrics[none_name] = metric[attributes.isna()]

        for u in sorted(unique_values):
            metrics[u] = metric[attributes == u]

        return metrics

    def _sort_vals_obj(self, vals: pd.Series, obj_fn: Optim) -> np.ndarray:
        if Optim(obj_fn) == Optim.min:
            return vals.sort_values(na_position="last").values
        return vals.sort_values(ascending=False, na_position="last").values


class ViolinPlot(Categorical):
    """
    Class for constructing violinplots. Its constructor takes in as input positional arguments or keyword
    arguments from the base class `Categorical`. Possible arguments are listed in the Parameters section.
    The Attributes section lists its own attributes as well as those that are inherited.

    Parameters
    ----------
    *args : ty.Any
        Positional arguments to pass to the base class ``Categorical``. These arguments are:

        - ``metric`` : ``pd.Series`` - The ablation study metric values to plot,

        - ``attributes`` : ``pd.Series`` - The ablation study attributes values to plot,

        - ``metric_obj_fn`` : ``Optim`` - The metric optimization direction,

        - ``y_axis`` : ``str, optional`` - The y-axis label (metric name), by default ``None``,

        - ``x_axis`` : ``str, optional`` - The x-axis label (attribute name), by default ``None``,

        - ``x_ticks`` : ``list[str], optional`` - The x-axis ticks, by default ``None``,

        - ``ax`` : ``Axes, optional`` - The axes to plot on, by default ``None``.

    **kwargs : ty.Any
        Keyword arguments to pass to the base class ``Categorical``. These arguments are:

        - ``metric`` : ``pd.Series`` - The ablation study metric values to plot,

        - ``attributes`` : ``pd.Series`` - The ablation study attributes values to plot,

        - ``metric_obj_fn`` : ``Optim`` - The metric optimization direction,

        - ``y_axis`` : ``str, optional`` - The y-axis label (metric name), by default ``None``,

        - ``x_axis`` : ``str, optional`` - The x-axis label (attribute name), by default ``None``,

        - ``x_ticks`` : ``list[str], optional`` - The x-axis ticks, by default ``None``,

        - ``ax`` : ``Axes, optional`` - The axes to plot on, by default ``None``.

    Attributes
    ----------
    metric : pd.Series
        The ablation study metric values to plot (with null value removed).
    attributes : pd.Series
        The ablation study attributes values to plot (with null metric value removed).
    metric_obj_fn : Optim
        The metric optimization direction.
    y_axis : str
        The y-axis label (metric name).
    x_axis : str
        The x-axis label (attribute name).
    x_ticks : list[str]
        The x-axis ticks.
    figure : Figure
        The figure to plot on. If `None`, a new figure of size `(4,4)` will be created.
    ax : Axes
        The axes to plot on. If `None`, a new axis will be created as the first subplot
        in the first cell and first column of a `1x1` grid.
    DATA_TYPE : str
        The attribute data type. In this case, it is `"categorical"`.
    attribute_metric_map : dict[str, pd.Series]
        A dictionary mapping attribute values to metric values.
    figsize: tuple
        A tuple representing the size of the figure in terms of axes `(x, y)`.

    """

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        sns.set()
        sns.set_style("whitegrid")
        self.figsize = (8, 4)
        super().__init__(*args, **kwargs)

    def _make_figure(self, ax: Axes | None = None) -> tuple[Figure | None, Axes]:
        figure = None
        if ax is None:
            figure = plt.figure(figsize=(10, 8))
            ax = figure.add_subplot(1, 1, 1)
        return figure, ax

    def _make(
        self,
        **kwargs: ty.Any,
    ) -> tuple[Figure, Axes]:
        sns.violinplot(
            [v.values for v in self.attribute_metric_map.values()],
            ax=self.ax,
            palette="Set3",
        )
        mean_perf = []
        std_perf = []
        median_perf = []
        best_perf = []
        for vals in self.attribute_metric_map.values():
            # top performance marker
            obj_fn = self.metrics_obj_fn
            best_perf.append(self._sort_vals_obj(vals, obj_fn)[0])

            std = np.std(vals)
            if Optim(obj_fn) == Optim.min:
                std *= -1
            mean_perf.append(np.mean(vals))
            std_perf.append(np.mean(vals) + std)
            median_perf.append(np.median(vals))

        labels = [
            f"Mean: {mean:.2e}\nBest: {best:.2e}\n{name}"
            for mean, best, name in zip(
                mean_perf, best_perf, self.attribute_metric_map.keys()
            )
        ]
        self.ax.set_xticks(
            np.arange(len(self.attribute_metric_map)),
            labels=labels,
        )

        sns.despine(left=True, bottom=True)
        return self.figure, self.ax

    def _parse_legend(self, ax: Axes):
        pass
