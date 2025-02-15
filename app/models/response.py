from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal


class BaseResponse(BaseModel):
    method: str
    success: bool
    result: Dict[str, Any]
    message: Optional[str] = None


class StatisticalTestResponse(BaseResponse):
    result: Dict[str, Any]
    plot_url: Optional[str] = None
    plot_base64: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    params: Optional[Dict[str, Any]] = None


class TTestResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "t_statistic": 0.0,
        "p_value": 0.0,
        "degrees_of_freedom": 0,
        "confidence_interval": [0.0, 0.0],
        "effect_size": 0.0,
        "conclusion": ""
    }


class HypothesisTestResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "z_statistic": 0.0,
        "p_value": 0.0,
        "confidence_interval": [0.0, 0.0],
        "conclusion": ""
    }


class LinearRegressionResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "coefficients": [0.0, 0.0],
        "r_squared": 0.0,
        "p_value": 0.0,
        "standard_error": 0.0,
        "equation": "",
        "predicted_y": Optional[float]
    }


class CorrelationAnalysisResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "correlation_coefficient": 0.0,
        "p_value": 0.0,
        "confidence_interval": [0.0, 0.0],
        "conclusion": ""
    }


class ChiSquareTestResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "chi_square_statistic": 0.0,
        "p_value": 0.0,
        "degrees_of_freedom": 0,
        "conclusion": ""
    }


class AnovaResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "f_statistic": 0.0,
        "p_value": 0.0,
        "eta_squared": 0.0,
        "group_statistics": [],
        "conclusion": ""
    }


class PairedTTestResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "t_statistic": 0.0,
        "p_value": 0.0,
        "mean_difference": 0.0,
        "std_difference": 0.0,
        "effect_size": 0.0,
        "confidence_interval": [0.0, 0.0],
        "conclusion": ""
    }


class DescriptiveResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "sample_size": 0,
        "mean": 0.0,
        "median": 0.0,
        "mode": {
            "value": 0.0,
            "count": 0
        },
        "std": 0.0,
        "variance": 0.0,
        "range": {
            "min": 0.0,
            "max": 0.0,
            "range": 0.0
        },
        "quartiles": {
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0
        },
        "shape": {
            "skewness": 0.0,
            "kurtosis": 0.0
        },
        "conclusion": ""
    }


class SurvivalAnalysisResponse(StatisticalTestResponse):
    result: Dict[str, Any] = {
        "median_survival": 0.0,
        "survival_probabilities": {},
        "confidence_intervals": {},
        "survival_plot": "",
        "hazard_plot": "",
        "log_rank_test": {
            "statistic": 0.0,
            "p_value": 0.0
        },
        "conclusion": ""
    }


class MultipleRegressionResponse(BaseResponse):
    method: Literal["multiple_regression"]
    success: bool
    result: Dict[str, Any]


class LogisticRegressionResponse(BaseResponse):
    method: Literal["logistic_regression"]
    success: bool
    result: Dict[str, Any] = {
        "model_summary": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "pseudo_r2": 0.0,
            "auc_roc": 0.0,
            "sample_size": 0,
            "feature_count": 0
        },
        "coefficients": [],
        "intercept": 0.0,
        "prediction": None,
        "prediction_probability": None,
        "plots": {
            "roc_plot": "",
            "coefficient_plot": "",
            "probability_plot": ""
        },
        "conclusion": ""
    }


class PCAResponse(BaseResponse):
    method: Literal["pca"]
    success: bool
    result: Dict[str, Any] = {
        "summary": {
            "n_components": 0,
            "n_features": 0,
            "n_samples": 0,
            "total_variance_explained": 0.0
        },
        "principal_components": [],
        "transformed_data": [],
        "feature_names": [],
        "plots": {
            "scree_plot": "",
            "cumulative_plot": "",
            "scatter_plot": None,
            "loading_plot": None,
            "heatmap_plot": ""
        },
        "conclusion": ""
    }


class FactorAnalysisResponse(BaseResponse):
    method: Literal["factor_analysis"]
    success: bool
    result: Dict[str, Any] = {
        "summary": {
            "n_factors": 0,
            "n_features": 0,
            "n_samples": 0,
            "total_variance_explained": 0.0,
            "rotation_method": ""
        },
        "factors": [],
        "communalities": {},
        "feature_names": [],
        "plots": {
            "scree_plot": "",
            "loading_plot": "",
            "loading_scatter": None
        },
        "conclusion": ""
    }
