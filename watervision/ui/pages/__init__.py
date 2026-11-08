"""One frame per module. The application shows a single frame at a time."""

from watervision.ui.pages.acquisition import AcquisitionPage
from watervision.ui.pages.dashboard import DashboardPage
from watervision.ui.pages.estimation import EstimationPage
from watervision.ui.pages.evaluation import EvaluationPage
from watervision.ui.pages.export_page import ExportPage
from watervision.ui.pages.fourier_page import FourierPage
from watervision.ui.pages.preprocessing import PreprocessingPage
from watervision.ui.pages.segmentation_page import SegmentationPage
from watervision.ui.pages.settings_page import SettingsPage

__all__ = [
    "AcquisitionPage",
    "DashboardPage",
    "EstimationPage",
    "EvaluationPage",
    "ExportPage",
    "FourierPage",
    "PreprocessingPage",
    "SegmentationPage",
    "SettingsPage",
]
