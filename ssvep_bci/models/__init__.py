"""
SSVEP BCI Models Package

Contains signal processing and classification components.
"""

from .eeg_buffer import EEGBuffer
from .preprocessor import SSVEPPreprocessor
from .cca_decoder import SSVEPDecoder, DecisionResult
from .calibration import (
    CalibrationCollector,
    CalibrationData,
    CalibrationSession,
    CalibrationTrial,
    FREQ_TO_EVENT_ID,
    EVENT_ID_TO_FREQ
)
from .template_cca import TemplateCCADecoder, TemplateCCAConfig

__all__ = [
    'EEGBuffer',
    'SSVEPPreprocessor',
    'SSVEPDecoder',
    'DecisionResult',
    'CalibrationCollector',
    'CalibrationData',
    'CalibrationSession',
    'CalibrationTrial',
    'FREQ_TO_EVENT_ID',
    'EVENT_ID_TO_FREQ',
    'TemplateCCADecoder',
    'TemplateCCAConfig',
]
