from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.figure import Figure


def get_calibration_curve(
    prediction: Sequence[int | float],
    label: Sequence[int | float],
    hue: Sequence[int] | None = None,
) -> Figure:
    """Generate calibration curve with matplotlib's pyplot

    Args:
        prediction (Sequence[int | float]): prediction vector
        label (Sequence[int | float]): ground truth vector
        hue (Sequence[int], optional): vector for hue purpose. Defaults to None.

    Returns:
        Figure: matplotlib's Figure object for the plot
    """
    sb.set_theme()
    fig = plt.figure(figsize=(6, 5))
    sb.scatterplot(x=label, y=prediction, hue=hue)
    min_lab = min(label)
    max_lab = max(label)
    plt.plot([min_lab, max_lab], [min_lab, max_lab], "r")
    plt.xlabel("True Motion Score")
    plt.ylabel("Estimated Motion Score")
    return fig
