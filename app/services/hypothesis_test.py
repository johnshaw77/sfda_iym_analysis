from typing import List, Dict, Any
import numpy as np
from scipy import stats
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array)
from ..utils.plotting import create_histogram, create_qq_plot
from ..utils.errors import CalculationError


def perform_hypothesis_test(data: List[float],
                            hypothesis_value: float) -> Dict[str, Any]:
    """
    執行單樣本 z 檢定
    
    參數:
        data: 樣本數據
        hypothesis_value: 虛無假設的值
        
    返回:
        包含檢定結果的字典
    """
    # 數據驗證
    validate_array_size(data, "data")
    validate_sample_size(data, "data")
    validate_numeric_array(data, "data")

    try:
        # 計算基本統計量
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        n = len(data)

        # 計算 z 統計量
        z_stat = (sample_mean - hypothesis_value) / (sample_std / np.sqrt(n))

        # 計算 p 值（雙尾檢定）
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # 計算置信區間 (95%)
        ci_margin = 1.96 * sample_std / np.sqrt(n)
        ci_lower = sample_mean - ci_margin
        ci_upper = sample_mean + ci_margin

        # 生成結論
        conclusion = "拒絕虛無假設" if p_value < 0.05 else "無法拒絕虛無假設"
        if p_value < 0.05:
            conclusion += "，樣本平均數與假設值有顯著差異"
        else:
            conclusion += "，未發現顯著差異"

        # 生成圖表
        hist_plot = create_histogram(data=data,
                                     title="數據分布直方圖",
                                     xlabel="數值",
                                     ylabel="頻率")

        qq_plot = create_qq_plot(data=data, title="Q-Q 圖")

        return {
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "sample_mean": float(sample_mean),
            "sample_std": float(sample_std),
            "confidence_interval": [float(ci_lower),
                                    float(ci_upper)],
            "conclusion": conclusion,
            "hist_plot": hist_plot,
            "qq_plot": qq_plot
        }

    except Exception as e:
        raise CalculationError(message=f"假設檢定計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
