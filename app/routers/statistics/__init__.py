from .basic_stats import router as basic_stats_router
from .correlation import router as correlation_router
from .regression import router as regression_router
from .survival import router as survival_router
from .dimension_reduction import router as dimension_reduction_router

__all__ = [
    "basic_stats_router",
    "correlation_router",
    "regression_router",
    "survival_router",
    "dimension_reduction_router"
] 