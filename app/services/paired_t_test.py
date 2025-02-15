from typing import List, Dict, Any
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError


def perform_paired_t_test(pre_test: List[float],
                          post_test: List[float]) -> Dict[str, Any]:
    """
    執行配對樣本 t 檢定
    
    參數:
        pre_test: 前測數據
        post_test: 後測數據
        
    返回:
        包含檢定結果的字典
    """
    # 數據驗證
    validate_array_size(pre_test, "pre_test")
    validate_array_size(post_test, "post_test")
    validate_sample_size(pre_test, "pre_test")
    validate_sample_size(post_test, "post_test")
    validate_numeric_array(pre_test, "pre_test")
    validate_numeric_array(post_test, "post_test")
    validate_equal_length(pre_test, post_test, "pre_test", "post_test")

    try:
        # 計算差異值
        differences = np.array(post_test) - np.array(pre_test)

        # 執行配對樣本 t 檢定
        t_stat, p_value = stats.ttest_rel(post_test, pre_test)

        # 計算效果量 (Cohen's d for paired samples)
        d = np.mean(differences) / np.std(differences, ddof=1)

        # 計算置信區間
        n = len(pre_test)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        t_critical = stats.t.ppf(0.975, n - 1)
        ci_margin = t_critical * (std_diff / np.sqrt(n))
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin

        # 生成結論
        conclusion = "拒絕虛無假設" if p_value < 0.05 else "無法拒絕虛無假設"
        if p_value < 0.05:
            conclusion += "，前後測有顯著差異"
            if mean_diff > 0:
                conclusion += "，後測顯著高於前測"
            else:
                conclusion += "，後測顯著低於前測"

            if abs(d) > 0.8:
                conclusion += "，效果量大"
            elif abs(d) > 0.5:
                conclusion += "，效果量中等"
            else:
                conclusion += "，效果量小"
        else:
            conclusion += "，前後測未發現顯著差異"

        # 生成箱型圖
        plt.figure(figsize=(10, 6))
        data = [pre_test, post_test]
        sns.boxplot(data=data)
        plt.xticks([0, 1], ['前測', '後測'])
        plt.title('前後測數據分布箱型圖')
        plt.xlabel('測試時間')
        plt.ylabel('數值')
        box_plot = get_base64_plot()

        # 生成差異值直方圖
        plt.figure(figsize=(10, 6))
        sns.histplot(differences, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('前後測差異值分布')
        plt.xlabel('差異值 (後測 - 前測)')
        plt.ylabel('頻率')
        diff_plot = get_base64_plot()

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "effect_size": float(d),
            "confidence_interval": [float(ci_lower),
                                    float(ci_upper)],
            "sample_size": n,
            "conclusion": conclusion,
            "box_plot": box_plot,
            "difference_plot": diff_plot
        }

    except Exception as e:
        raise CalculationError(message=f"配對樣本 t 檢定計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
