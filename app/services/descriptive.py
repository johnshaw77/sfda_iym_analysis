from typing import List, Dict, Any
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError, ValidationError


def perform_descriptive_analysis(data: List[float]) -> Dict[str, Any]:
    """
    執行描述性統計分析
    
    參數:
        data: 數據列表
        
    返回:
        包含分析結果的字典
    """
    try:
        # 數據驗證
        if not data:
            raise ValidationError("數據不能為空")

        validate_array_size(data, "data")
        validate_sample_size(data, "data")
        validate_numeric_array(data, "data")

        # 轉換為 numpy 數組
        arr = np.array(data, dtype=float)

        # 檢查是否包含 NaN 或無限值
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValidationError("數據包含無效值（NaN 或無限值）")

        # 計算基本統計量
        n = len(data)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1))
        var = float(np.var(arr, ddof=1))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        range_val = max_val - min_val

        # 計算四分位數
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1

        # 計算偏度和峰度
        skewness = float(stats.skew(arr))
        kurtosis = float(stats.kurtosis(arr))

        # 計算眾數（使用 numpy 的 unique 函數）
        unique_vals, counts = np.unique(arr, return_counts=True)
        mode_idx = np.argmax(counts)
        mode = float(unique_vals[mode_idx])
        mode_count = int(counts[mode_idx])

        # 生成描述性結論
        distribution_type = "對稱" if abs(skewness) < 0.5 else (
            "右偏" if skewness > 0 else "左偏")
        peak_type = "常態" if abs(kurtosis) < 0.5 else (
            "尖峰" if kurtosis > 0 else "平峰")

        conclusion = f"數據分布呈現{distribution_type}分布，{peak_type}特徵"
        if iqr / range_val > 0.5:
            conclusion += "，數據分散程度較大"
        else:
            conclusion += "，數據相對集中"

        # 設置中文字體
        try:
            plt.rcParams['font.family'] = ['PingFang HK']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            try:
                plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                print("無法載入中文字體，將使用預設字體")

        # 生成直方圖和核密度圖
        plt.figure(figsize=(10, 6))
        sns.histplot(data=arr, kde=True)
        plt.axvline(x=mean, color='r', linestyle='--', label='平均數')
        plt.axvline(x=median, color='g', linestyle='--', label='中位數')
        plt.title('數據分布圖')
        plt.xlabel('數值')
        plt.ylabel('頻率')
        plt.legend()
        dist_plot = get_base64_plot()

        # 生成箱型圖
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=arr)
        plt.title('箱型圖')
        plt.ylabel('數值')
        box_plot = get_base64_plot()

        # 生成 Q-Q 圖
        plt.figure(figsize=(10, 6))
        stats.probplot(arr, dist="norm", plot=plt)
        plt.title('Q-Q 圖')
        qq_plot = get_base64_plot()

        return {
            "sample_size": n,
            "mean": mean,
            "median": median,
            "mode": {
                "value": mode,
                "count": mode_count
            },
            "std": std,
            "variance": var,
            "range": {
                "min": min_val,
                "max": max_val,
                "range": range_val
            },
            "quartiles": {
                "q1": q1,
                "q3": q3,
                "iqr": iqr
            },
            "shape": {
                "skewness": skewness,
                "kurtosis": kurtosis
            },
            "conclusion": conclusion,
            "distribution_plot": dist_plot,
            "box_plot": box_plot,
            "qq_plot": qq_plot
        }

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"描述性統計分析過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
