from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .utils.errors import StatisticalError
from .models.method_info import get_available_methods, StatisticalMethodInfo
from .routers.statistics import (
    basic_stats_router,
    regression_router,
    correlation_router,
    dimension_reduction_router,
    survival_router
)
import time
from typing import List, Dict
from fastapi.routing import APIRoute

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
                        content={
                            "detail": exc.message,
                            "error_code": exc.error_code,
                            "params": exc.params
                        })


@app.get("/")
async def root():
    """首頁"""
    return FileResponse('static/index.html')


@app.get("/static/stat_method.html")
async def stat_method():
    """統計方法說明頁面"""
    return FileResponse('static/stat_method.html')


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


# 註冊路由
app.include_router(basic_stats_router, prefix="/api/v1/statistics", tags=["基本統計"])
app.include_router(regression_router, prefix="/api/v1/statistics", tags=["回歸分析"])
app.include_router(correlation_router, prefix="/api/v1/statistics", tags=["相關分析"])
app.include_router(dimension_reduction_router, prefix="/api/v1/statistics", tags=["降維分析"])
app.include_router(survival_router, prefix="/api/v1/statistics", tags=["存活分析"])

@app.get("/api/v1/statistics/routes", tags=["API資訊"])
async def get_statistics_routes() -> Dict[str, List[Dict[str, str]]]:
    """
    獲取所有統計相關的API路徑
    
    Returns:
        Dict[str, List[Dict[str, str]]]: 按標籤分組的API路徑列表
    """
    statistics_routes = {}
    
    # 獲取所有路由
    routes = app.routes
    
    # 過濾並組織統計相關的路由
    for route in routes:
        if isinstance(route, APIRoute) and route.tags:  # 確保路由有標籤
            for tag in route.tags:
                if "分析" in tag or "統計" in tag:  # 篩選統計相關的路由
                    if tag not in statistics_routes:
                        statistics_routes[tag] = []
                    
                    statistics_routes[tag].append({
                        "path": route.path,
                        "method": route.methods.pop() if route.methods else "GET",  # 獲取HTTP方法
                        "summary": route.summary or "無描述",
                        "operation_id": route.name
                    })
    
    return statistics_routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
