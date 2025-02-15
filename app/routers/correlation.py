from fastapi import APIRouter, HTTPException
from ..models.request import CorrelationAnalysisRequest
from ..models.response import CorrelationAnalysisResponse, ErrorResponse
from ..services.correlation import perform_correlation_analysis
from ..utils.errors import StatisticalError, ValidationError

router = APIRouter(prefix="/api/v1/analyze", tags=["相關性分析"])


@router.post("/correlation", response_model=CorrelationAnalysisResponse)
async def correlation_analysis(request: CorrelationAnalysisRequest):
    try:
        x_values = request.data.get("x", [])
        y_values = request.data.get("y", [])
        if not x_values or not y_values:
            raise ValidationError("缺少必要的數據：x 或 y")
        result = perform_correlation_analysis(x_values, y_values)
        return {
            "method": "correlation_analysis",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)
