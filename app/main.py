from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .utils.errors import StatisticalError
from .models.method_info import get_available_methods, StatisticalMethodInfo
from .routers import basic_stats, regression, correlation, dimension_reduction, survival
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


# 引入所有路由
app.include_router(basic_stats.router)
app.include_router(regression.router)
app.include_router(correlation.router)
app.include_router(dimension_reduction.router)
app.include_router(survival.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
