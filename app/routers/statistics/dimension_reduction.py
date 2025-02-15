from fastapi import APIRouter, HTTPException
from ...models.request import PCARequest, FactorAnalysisRequest
from ...models.response import PCAResponse, FactorAnalysisResponse, ErrorResponse
from ...services.pca import perform_pca
from ...services.factor_analysis import perform_factor_analysis
from ...utils.errors import StatisticalError, ValidationError

router = APIRouter(tags=["降維分析"])


@router.post("/pca", response_model=PCAResponse)
async def principal_component_analysis(request: PCARequest):
    try:
        data = request.data.get("data", [])
        n_components = request.data.get("n_components")
        if not data:
            raise ValidationError("缺少必要的數據：data")
        result = perform_pca(data, n_components)
        return {"method": "pca", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/factor-analysis", response_model=FactorAnalysisResponse)
async def factor_analysis(request: FactorAnalysisRequest):
    try:
        data = request.data.get("data", [])
        n_factors = request.data.get("n_factors")
        rotation = request.data.get("rotation", "varimax")
        if not data:
            raise ValidationError("缺少必要的數據：data")
        result = perform_factor_analysis(data, n_factors, rotation)
        return {"method": "factor_analysis", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)
