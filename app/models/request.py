from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Literal
from numpy import float64


class BaseRequest(BaseModel):
    method: Literal["t_test", "hypothesis_test", "linear_regression",
                    "correlation_analysis", "chi_square_test", "anova",
                    "paired_t_test", "descriptive", "survival_analysis",
                    "multiple_regression", "logistic_regression", "pca",
                    "factor_analysis"]
    data: dict


class TTestRequest(BaseRequest):
    method: Literal["t_test"]
    data: dict = Field(
        ..., example={
            "group1": [1.2, 2.3, 3.1],
            "group2": [1.8, 2.5, 3.0]
        })


class HypothesisTestRequest(BaseRequest):
    method: Literal["hypothesis_test"]
    data: dict = Field(
        ..., example={
            "data": [1.2, 2.3, 3.1],
            "hypothesis_value": 2.0
        })


class LinearRegressionRequest(BaseRequest):
    method: Literal["linear_regression"]
    data: dict = Field(
        ..., example={
            "x": [1, 2, 3],
            "y": [2, 4, 6],
            "predict_x": 4
        })


class CorrelationAnalysisRequest(BaseRequest):
    method: Literal["correlation_analysis"]
    data: dict = Field(..., example={"x": [1, 2, 3], "y": [2, 4, 6]})


class ChiSquareTestRequest(BaseRequest):
    method: Literal["chi_square_test"]
    data: dict = Field(
        ..., example={
            "observed": [10, 20, 30],
            "expected": [15, 25, 35]
        })


class AnovaRequest(BaseRequest):
    method: Literal["anova"]
    data: dict = Field(
        ...,
        example={
            "groups": [[1.2, 2.3, 3.1], [1.8, 2.5, 3.0], [2.1, 2.8, 3.3]],
            "group_names": ["組別1", "組別2", "組別3"]
        })


class PairedTTestRequest(BaseRequest):
    method: Literal["paired_t_test"]
    data: dict = Field(
        ...,
        example={
            "pre_test": [1.2, 2.3, 3.1],
            "post_test": [1.8, 2.5, 3.0]
        })


class DescriptiveRequest(BaseRequest):
    method: Literal["descriptive"]
    data: dict = Field(..., example={"data": [1.2, 2.3, 3.1, 2.8, 2.5, 3.0]})


class SurvivalAnalysisRequest(BaseRequest):
    method: Literal["survival_analysis"]
    data: dict = Field(
        ...,
        example={
            "durations": [10.2, 15.3, 20.1, 25.4, 30.2],
            "events": [1, 1, 0, 1, 0],
            "groups": [1, 1, 2, 2, 2],
            "group_names": ["治療組", "對照組"]
        })


class MultipleRegressionRequest(BaseRequest):
    method: Literal["multiple_regression"]
    data: dict = Field(
        ...,
        example={
            "X": [[1.0, 2.0, 3.0], [2.1, 3.2, 4.1], [0.5, 1.5, 2.5]],
            "y": [10.2, 15.3, 20.1],
            "feature_names": ["特徵1", "特徵2", "特徵3"],
            "predict_X": [2.5, 3.5, 1.5]
        })


class LogisticRegressionRequest(BaseRequest):
    method: Literal["logistic_regression"]
    data: dict = Field(
        ...,
        example={
            "X": [[1.0, 2.0, 3.0], [2.1, 3.2, 4.1], [0.5, 1.5, 2.5]],
            "y": [0, 1, 0, 1, 1],
            "feature_names": ["年齡", "收入", "教育程度"],
            "predict_X": [2.5, 3.5, 1.5]
        })


class PCARequest(BaseRequest):
    method: Literal["pca"]
    data: dict = Field(
        ...,
        example={
            "X": [
                [1.2, 2.3, 3.1, 2.8, 2.5, 3.0],  # 特徵1的數據
                [2.1, 3.2, 2.9, 3.1, 2.8, 3.3],  # 特徵2的數據
                [1.8, 2.5, 3.0, 2.7, 2.4, 2.9],  # 特徵3的數據
                [2.4, 3.1, 2.8, 3.0, 2.6, 3.2]  # 特徵4的數據
            ],
            "feature_names": ["身高", "體重", "年齡", "收入"],
            "n_components":
            2  # 可選，要保留的主成分數量
        })


class FactorAnalysisRequest(BaseRequest):
    method: Literal["factor_analysis"]
    data: dict = Field(
        ...,
        example={
            "X": [
                [1.2, 2.3, 3.1, 2.8, 2.5, 3.0],  # 特徵1的數據
                [2.1, 3.2, 2.9, 3.1, 2.8, 3.3],  # 特徵2的數據
                [1.8, 2.5, 3.0, 2.7, 2.4, 2.9],  # 特徵3的數據
                [2.4, 3.1, 2.8, 3.0, 2.6, 3.2]  # 特徵4的數據
            ],
            "feature_names": ["智力測驗", "學習動機", "學習態度", "學習成績"],
            "n_factors":
            2,  # 可選，要提取的因子數量
            "rotation":
            "varimax"  # 可選，旋轉方法
        })
