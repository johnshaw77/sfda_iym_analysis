from fastapi import HTTPException
from typing import Any, Dict, Optional


class StatisticalError(Exception):

    def __init__(self,
                 message: str,
                 error_code: str,
                 status_code: int = 400,
                 params: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.params = params or {}
        super().__init__(self.message)


class ValidationError(StatisticalError):

    def __init__(self, message: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(message=message,
                         error_code="VALIDATION_ERROR",
                         status_code=400,
                         params=params)


class DataSizeError(StatisticalError):

    def __init__(self, message: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(message=message,
                         error_code="DATA_SIZE_ERROR",
                         status_code=400,
                         params=params)


class CalculationError(StatisticalError):

    def __init__(self, message: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(message=message,
                         error_code="CALCULATION_ERROR",
                         status_code=500,
                         params=params)
