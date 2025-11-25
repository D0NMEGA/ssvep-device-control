"""
GUI Widgets for SSVEP BCI

This module exports all widget classes.
"""

from .signal_plot import SignalPlotWidget
from .classification_panel import ClassificationPanel, CorrelationBar
from .control_panel import ControlPanel

__all__ = [
    'SignalPlotWidget',
    'ClassificationPanel',
    'CorrelationBar',
    'ControlPanel',
]
