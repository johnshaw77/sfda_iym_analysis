from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import create_box_plot
from ..utils.errors import CalculationError


def perform_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """
    執行獨立樣本 t 檢定
    
    參數:
        group1: 第一組數據
        group2: 第二組數據
        
    返回:
        包含檢定結果的字典
    """
    # 數據驗證
    validate_array_size(group1, "group1")
    validate_array_size(group2, "group2")
    validate_sample_size(group1, "group1")
    validate_sample_size(group2, "group2")
    validate_numeric_array(group1, "group1")
    validate_numeric_array(group2, "group2")

    try:
        # 執行 t 檢定
        t_stat, p_value = stats.ttest_ind(group1, group2)

        # 計算效果量 (Cohen's d)
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_se

        # 計算置信區間
        mean_diff = np.mean(group1) - np.mean(group2)
        degrees_of_freedom = n1 + n2 - 2
        t_critical = stats.t.ppf(0.975, degrees_of_freedom)
        margin_of_error = t_critical * np.sqrt(var1 / n1 + var2 / n2)
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error

        # 生成結論
        conclusion = "拒絕虛無假設" if p_value < 0.05 else "無法拒絕虛無假設"
        if p_value < 0.05:
            conclusion += "，兩組數據有顯著差異"
        else:
            conclusion += "，未發現顯著差異"

        # 生成圖表
        plot_base64 = create_box_plot(data=[group1, group2],
                                      labels=["組別1", "組別2"],
                                      title="t 檢定箱型圖比較",
                                      xlabel="組別",
                                      ylabel="數值")

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(degrees_of_freedom),
            "confidence_interval": [float(ci_lower),
                                    float(ci_upper)],
            "effect_size": float(effect_size),
            "conclusion": conclusion,
            "plot_base64": plot_base64
        }

    except Exception as e:
        raise CalculationError(message=f"t 檢定計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
