from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API 設置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "統計分析 API 服務"

    # 安全設置
    SECRET_KEY: str = "your-secret-key-here"  # 在生產環境中應該從環境變數讀取
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # 速率限制
    RATE_LIMIT_PER_MINUTE: int = 60

    # 數據限制
    MAX_ARRAY_SIZE: int = 10000  # 最大數組大小

    # 數據驗證
    MIN_SAMPLE_SIZE: int = 3  # 最小樣本大小

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
