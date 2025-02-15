from fastapi import APIRouter, HTTPException
from ..models.request import (DescriptiveRequest, HypothesisTestRequest,
                              TTestRequest, PairedTTestRequest,
                              ChiSquareTestRequest, AnovaRequest)
from ..models.response import (DescriptiveResponse, HypothesisTestResponse,
                               TTestResponse, PairedTTestResponse,
                               ChiSquareTestResponse, AnovaResponse,
                               ErrorResponse)
from ..services.descriptive import perform_descriptive_analysis
from ..services.hypothesis_test import perform_hypothesis_test
from ..services.t_test import perform_t_test
from ..services.paired_t_test import perform_paired_t_test
from ..services.chi_square import perform_chi_square_test
from ..services.anova import perform_anova
from ..utils.errors import StatisticalError, ValidationError

router = APIRouter(prefix="/api/v1/analyze", tags=["基礎統計分析"])


@router.post("/descriptive", response_model=DescriptiveResponse)
async def descriptive_analysis(request: DescriptiveRequest):
    try:
        data = request.data.get("data", [])
        if not data:
            raise ValidationError("缺少必要的數據：data")
        result = perform_descriptive_analysis(data)
        return {"method": "descriptive", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/t-test", response_model=TTestResponse)
async def t_test(request: TTestRequest):
    try:
        group1 = request.data.get("group1", [])
        group2 = request.data.get("group2", [])
        if not group1 or not group2:
            raise ValidationError("缺少必要的數據：group1 或 group2")
        result = perform_t_test(group1, group2)
        return {"method": "t_test", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/hypothesis-test", response_model=HypothesisTestResponse)
async def hypothesis_test(request: HypothesisTestRequest):
    try:
        data = request.data.get("data", [])
        hypothesis_value = request.data.get("hypothesis_value")
        if not data or hypothesis_value is None:
            raise ValidationError("缺少必要的數據：data 或 hypothesis_value")
        result = perform_hypothesis_test(data, float(hypothesis_value))
        return {"method": "hypothesis_test", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/paired-t-test", response_model=PairedTTestResponse)
async def paired_t_test(request: PairedTTestRequest):
    try:
        pre_test = request.data.get("pre_test", [])
        post_test = request.data.get("post_test", [])
        if not pre_test or not post_test:
            raise ValidationError("缺少必要的數據：pre_test 或 post_test")
        result = perform_paired_t_test(pre_test, post_test)
        return {"method": "paired_t_test", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/chi-square", response_model=ChiSquareTestResponse)
async def chi_square_test(request: ChiSquareTestRequest):
    try:
        observed = request.data.get("observed", [])
        expected = request.data.get("expected", [])
        if not observed or not expected:
            raise ValidationError("缺少必要的數據：observed 或 expected")
        result = perform_chi_square_test(observed, expected)
        return {"method": "chi_square_test", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@router.post("/anova", response_model=AnovaResponse)
async def anova_test(request: AnovaRequest):
    try:
        groups = request.data.get("groups", [])
        group_names = request.data.get("group_names")
        if not groups:
            raise ValidationError("缺少必要的數據：groups")
        result = perform_anova(groups, group_names)
        return {"method": "anova", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)
