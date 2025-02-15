from fastapi import APIRouter, HTTPException
from ..models.request import (LinearRegressionRequest,
                              MultipleRegressionRequest,
                              LogisticRegressionRequest)
from ..models.response import (LinearRegressionResponse,
                               MultipleRegressionResponse,
                               LogisticRegressionResponse, ErrorResponse)
from ..services.linear_regression import perform_linear_regression
from ..services.multiple_regression import perform_multiple_regression
from ..services.logistic_regression import perform_logistic_regression
from ..utils.errors import StatisticalError, ValidationError

router = APIRouter(prefix="/api/v1/analyze", tags=["回歸分析"])


@router.post("/linear-regression", response_model=LinearRegressionResponse)
async def linear_regression(request: LinearRegressionRequest):
    try:
        x_values = request.data.get("x", [])
        y_values = request.data.get("y", [])
        predict_x = request.data.get("predict_x")
        if not x_values or not y_values:
            raise ValidationError("缺少必要的數據：x 或 y")
        result = perform_linear_regression(x_values, y_values, predict_x)
        return {
            "method": "linear_regression",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/multiple-regression", response_model=MultipleRegressionResponse)
async def multiple_regression(request: MultipleRegressionRequest):
    try:
        X = request.data.get("X", [])
        y = request.data.get("y", [])
        feature_names = request.data.get("feature_names", [])
        if not X or not y:
            raise ValidationError("缺少必要的數據：X 或 y")
        result = perform_multiple_regression(X, y, feature_names)
        return {
            "method": "multiple_regression",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/logistic-regression", response_model=LogisticRegressionResponse)
async def logistic_regression(request: LogisticRegressionRequest):
    try:
        X = request.data.get("X", [])
        y = request.data.get("y", [])
        feature_names = request.data.get("feature_names", [])
        if not X or not y:
            raise ValidationError("缺少必要的數據：X 或 y")
        result = perform_logistic_regression(X, y, feature_names)
        return {
            "method": "logistic_regression",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)
