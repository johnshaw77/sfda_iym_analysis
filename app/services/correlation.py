from typing import List, Dict, Any
import numpy as np
from scipy import stats
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import create_scatter_plot
from ..utils.errors import CalculationError


def perform_correlation_analysis(x: List[float],
                                 y: List[float]) -> Dict[str, Any]:
    """
    執行相關性分析
    
    參數:
        x: 第一個變量數據
        y: 第二個變量數據
        
    返回:
        包含分析結果的字典
    """
    # 數據驗證
    validate_array_size(x, "x")
    validate_array_size(y, "y")
    validate_sample_size(x, "x")
    validate_sample_size(y, "y")
    validate_numeric_array(x, "x")
    validate_numeric_array(y, "y")
    validate_equal_length(x, y, "x", "y")

    try:
        # 計算 Pearson 相關係數
        correlation_coef, p_value = stats.pearsonr(x, y)

        # 計算 Spearman 等級相關係數
        spearman_coef, spearman_p = stats.spearmanr(x, y)

        # 計算置信區間
        r = correlation_coef
        n = len(x)
        fisher_z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        ci_lower = np.tanh(fisher_z - 1.96 * se)
        ci_upper = np.tanh(fisher_z + 1.96 * se)

        # 生成結論
        conclusion = "存在顯著相關性" if p_value < 0.05 else "不存在顯著相關性"
        if p_value < 0.05:
            if correlation_coef > 0:
                conclusion += "，呈現正相關"
            else:
                conclusion += "，呈現負相關"

            if abs(correlation_coef) > 0.7:
                conclusion += "，相關程度強"
            elif abs(correlation_coef) > 0.3:
                conclusion += "，相關程度中等"
            else:
                conclusion += "，相關程度弱"

        # 生成散點圖
        plot_base64 = create_scatter_plot(x=x,
                                          y=y,
                                          title="相關性散點圖",
                                          xlabel="X 變量",
                                          ylabel="Y 變量")

        return {
            "correlation_coefficient": float(correlation_coef),
            "p_value": float(p_value),
            "spearman_coefficient": float(spearman_coef),
            "spearman_p_value": float(spearman_p),
            "confidence_interval": [float(ci_lower),
                                    float(ci_upper)],
            "sample_size": n,
            "conclusion": conclusion,
            "plot_base64": plot_base64
        }

    except Exception as e:
        raise CalculationError(message=f"相關性分析計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
