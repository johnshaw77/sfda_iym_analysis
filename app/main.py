from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .config import settings
from .models.request import (
    TTestRequest, HypothesisTestRequest, LinearRegressionRequest,
    CorrelationAnalysisRequest, ChiSquareTestRequest, AnovaRequest,
    PairedTTestRequest, DescriptiveRequest, SurvivalAnalysisRequest,
    MultipleRegressionRequest, LogisticRegressionRequest, PCARequest,
    FactorAnalysisRequest)
from .models.response import (
    TTestResponse, HypothesisTestResponse, LinearRegressionResponse,
    CorrelationAnalysisResponse, ChiSquareTestResponse, AnovaResponse,
    PairedTTestResponse, DescriptiveResponse, SurvivalAnalysisResponse,
    ErrorResponse, MultipleRegressionResponse, LogisticRegressionResponse,
    PCAResponse, FactorAnalysisResponse)
from .services.t_test import perform_t_test
from .services.linear_regression import perform_linear_regression
from .services.hypothesis_test import perform_hypothesis_test
from .services.correlation import perform_correlation_analysis
from .services.chi_square import perform_chi_square_test
from .services.anova import perform_anova
from .services.paired_t_test import perform_paired_t_test
from .services.descriptive import perform_descriptive_analysis
from .services.survival import perform_survival_analysis
from .services.multiple_regression import perform_multiple_regression
from .services.logistic_regression import perform_logistic_regression
from .services.pca import perform_pca
from .services.factor_analysis import perform_factor_analysis
from .utils.errors import StatisticalError, ValidationError
from app.models.method_info import get_available_methods, StatisticalMethodInfo
import time
from typing import List

app = FastAPI(title=settings.PROJECT_NAME,
              openapi_url=f"{settings.API_V1_STR}/openapi.json")

# 添加靜態文件支持
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 請求計數器
request_counts = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # 獲取客戶端 IP
    client_ip = request.client.host
    current_time = time.time()

    # 清理舊的請求記錄
    request_counts[client_ip] = [
        timestamp for timestamp in request_counts.get(client_ip, [])
        if current_time - timestamp < 60
    ]

    # 檢查請求頻率
    if len(request_counts.get(client_ip,
                              [])) >= settings.RATE_LIMIT_PER_MINUTE:
        return JSONResponse(status_code=429,
                            content={"detail": "請求過於頻繁，請稍後再試"})

    # 添加新的請求記錄
    request_counts.setdefault(client_ip, []).append(current_time)

    response = await call_next(request)
    return response


@app.exception_handler(StatisticalError)
async def statistical_error_handler(request: Request, exc: StatisticalError):
    return JSONResponse(status_code=exc.status_code,
                        content=ErrorResponse(detail=exc.message,
                                              error_code=exc.error_code,
                                              params=exc.params).dict())


@app.get("/")
async def root():
    return FileResponse('static/index.html')


@app.get("/static/stat_method.html")
async def stat_method():
    return FileResponse('static/stat_method.html')


@app.post("/api/v1/analyze/t-test", response_model=TTestResponse)
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


@app.post("/api/v1/analyze/hypothesis-test",
          response_model=HypothesisTestResponse)
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


@app.post("/api/v1/analyze/linear-regression",
          response_model=LinearRegressionResponse)
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


@app.post("/api/v1/analyze/correlation",
          response_model=CorrelationAnalysisResponse)
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


@app.post("/api/v1/analyze/correlation_analysis",
          response_model=CorrelationAnalysisResponse)
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


@app.post("/api/v1/analyze/chi-square", response_model=ChiSquareTestResponse)
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


@app.post("/api/v1/analyze/anova", response_model=AnovaResponse)
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


@app.post("/api/v1/analyze/paired-t-test", response_model=PairedTTestResponse)
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


@app.post("/api/v1/analyze/descriptive", response_model=DescriptiveResponse)
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


@app.post("/api/v1/analyze/survival", response_model=SurvivalAnalysisResponse)
async def survival_analysis(request: SurvivalAnalysisRequest):
    try:
        durations = request.data.get("durations", [])
        events = request.data.get("events", [])
        groups = request.data.get("groups")
        group_names = request.data.get("group_names")

        if not durations or not events:
            raise ValidationError("缺少必要的數據：durations 或 events")

        result = perform_survival_analysis(durations, events, groups,
                                           group_names)
        return {
            "method": "survival_analysis",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@app.post("/api/v1/analyze/regression",
          response_model=LinearRegressionResponse)
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


@app.post("/api/v1/analyze/multiple-regression",
          response_model=MultipleRegressionResponse)
async def multiple_regression(request: MultipleRegressionRequest):
    try:
        X = request.data.get("X", [])
        y = request.data.get("y", [])
        feature_names = request.data.get("feature_names")
        predict_X = request.data.get("predict_X")

        if not X or not y:
            raise ValidationError("缺少必要的數據：X 或 y")

        result = perform_multiple_regression(X, y, feature_names, predict_X)
        return {
            "method": "multiple_regression",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@app.post("/api/v1/analyze/logistic-regression",
          response_model=LogisticRegressionResponse)
async def logistic_regression(request: LogisticRegressionRequest):
    try:
        X = request.data.get("X", [])
        y = request.data.get("y", [])
        feature_names = request.data.get("feature_names")
        predict_X = request.data.get("predict_X")

        if not X or not y:
            raise ValidationError("缺少必要的數據：X 或 y")

        result = perform_logistic_regression(X, y, feature_names, predict_X)
        return {
            "method": "logistic_regression",
            "success": True,
            "result": result
        }
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@app.post("/api/v1/analyze/pca", response_model=PCAResponse)
async def principal_component_analysis(request: PCARequest):
    try:
        X = request.data.get("X", [])
        feature_names = request.data.get("feature_names")
        n_components = request.data.get("n_components")

        if not X:
            raise ValidationError("缺少必要的數據：X")

        result = perform_pca(X, feature_names, n_components)
        return {"method": "pca", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@app.post("/api/v1/analyze/factor-analysis",
          response_model=FactorAnalysisResponse)
async def factor_analysis(request: FactorAnalysisRequest):
    try:
        X = request.data.get("X", [])
        feature_names = request.data.get("feature_names")
        n_factors = request.data.get("n_factors")
        rotation = request.data.get("rotation", "varimax")

        if not X:
            raise ValidationError("缺少必要的數據：X")

        result = perform_factor_analysis(X, feature_names, n_factors, rotation)
        return {"method": "factor_analysis", "success": True, "result": result}
    except Exception as e:
        raise StatisticalError(message=str(e),
                               error_code="CALCULATION_ERROR",
                               status_code=400)


@app.get("/api/v1/methods",
         response_model=List[StatisticalMethodInfo],
         tags=["統計方法"])
async def list_statistical_methods():
    """
    列出所有可用的統計分析方法及其使用情境說明
    
    Returns:
        List[StatisticalMethodInfo]: 統計方法資訊列表
    """
    return get_available_methods()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
