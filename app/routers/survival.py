from fastapi import APIRouter, HTTPException
from ..models.request import SurvivalAnalysisRequest
from ..models.response import SurvivalAnalysisResponse, ErrorResponse
from ..services.survival import perform_survival_analysis
from ..utils.errors import StatisticalError, ValidationError

router = APIRouter(prefix="/api/v1/analyze", tags=["存活分析"])


@router.post("/survival", response_model=SurvivalAnalysisResponse)
async def survival_analysis(request: SurvivalAnalysisRequest):
    try:
        times = request.data.get("times", [])
        events = request.data.get("events", [])
        groups = request.data.get("groups", [])
        group_names = request.data.get("group_names", [])
        if not times or not events:
            raise ValidationError("缺少必要的數據：times 或 events")
        result = perform_survival_analysis(times, events, groups, group_names)
        return {
            "method": "survival_analysis",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)
