from typing import List, Optional, Union, Any
import numpy as np
from .errors import ValidationError, DataSizeError
from ..config import settings


def validate_array_size(data: Union[List[float], np.ndarray],
                        name: str) -> None:
    """驗證數組大小是否在允許範圍內"""
    if len(data) > settings.MAX_ARRAY_SIZE:
        raise DataSizeError(
            message=f"{name} 數組大小超過限制 {settings.MAX_ARRAY_SIZE}",
            params={
                "array_name": name,
                "size": len(data)
            })


def validate_sample_size(data: Union[List[float], np.ndarray],
                         name: str) -> None:
    """驗證樣本大小是否足夠"""
    if len(data) < settings.MIN_SAMPLE_SIZE:
        raise ValidationError(
            message=f"{name} 樣本大小不足，最少需要 {settings.MIN_SAMPLE_SIZE} 個數據點",
            params={
                "array_name": name,
                "size": len(data)
            })


def validate_equal_length(data1: Union[List[float], np.ndarray],
                          data2: Union[List[float], np.ndarray], name1: str,
                          name2: str) -> None:
    """驗證兩個數組長度是否相等"""
    if len(data1) != len(data2):
        raise ValidationError(message=f"{name1} 和 {name2} 的長度不相等",
                              params={
                                  "array1_name": name1,
                                  "array1_size": len(data1),
                                  "array2_name": name2,
                                  "array2_size": len(data2)
                              })


def validate_numeric_array(data: Union[List[Any], np.ndarray],
                           name: str) -> None:
    """驗證數組是否為數值型"""
    try:
        np.array(data, dtype=float)
    except (ValueError, TypeError):
        raise ValidationError(message=f"{name} 包含非數值型數據",
                              params={"array_name": name})


def validate_positive_array(data: Union[List[float], np.ndarray],
                            name: str) -> None:
    """驗證數組是否全為正數"""
    if not np.all(np.array(data) > 0):
        raise ValidationError(message=f"{name} 包含非正數",
                              params={"array_name": name})


def validate_non_negative_array(data: Union[List[float], np.ndarray],
                                name: str) -> None:
    """驗證數組是否全為非負數"""
    if not np.all(np.array(data) >= 0):
        raise ValidationError(message=f"{name} 包含負數",
                              params={"array_name": name})
