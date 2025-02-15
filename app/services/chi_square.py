from typing import List, Dict, Any
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length,
                                validate_non_negative_array)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError


def perform_chi_square_test(observed: List[float],
                            expected: List[float]) -> Dict[str, Any]:
    """
    執行卡方檢定
    
    參數:
        observed: 觀察值
        expected: 期望值
        
    返回:
        包含檢定結果的字典
    """
    # 數據驗證
    validate_array_size(observed, "observed")
    validate_array_size(expected, "expected")
    validate_sample_size(observed, "observed")
    validate_sample_size(expected, "expected")
    validate_numeric_array(observed, "observed")
    validate_numeric_array(expected, "expected")
    validate_equal_length(observed, expected, "observed", "expected")
    validate_non_negative_array(observed, "observed")
    validate_non_negative_array(expected, "expected")

    try:
        # 執行卡方檢定
        chi2_stat, p_value = stats.chisquare(observed, expected)

        # 計算自由度
        df = len(observed) - 1

        # 計算各類別的貢獻
        contributions = (np.array(observed) -
                         np.array(expected))**2 / np.array(expected)

        # 生成結論
        conclusion = "拒絕虛無假設" if p_value < 0.05 else "無法拒絕虛無假設"
        if p_value < 0.05:
            conclusion += "，觀察值與期望值有顯著差異"
        else:
            conclusion += "，未發現顯著差異"

        # 生成比較圖
        plt.figure(figsize=(10, 6))
        categories = range(len(observed))
        width = 0.35

        plt.bar([x - width / 2 for x in categories],
                observed,
                width,
                label='觀察值',
                color='skyblue')
        plt.bar([x + width / 2 for x in categories],
                expected,
                width,
                label='期望值',
                color='lightgreen')

        plt.xlabel('類別')
        plt.ylabel('頻率')
        plt.title('觀察值與期望值比較')
        plt.legend()
        plt.xticks(categories)

        plot_base64 = get_base64_plot()

        # 生成貢獻圖
        plt.figure(figsize=(10, 6))
        plt.bar(categories, contributions, color='salmon')
        plt.xlabel('類別')
        plt.ylabel('卡方貢獻值')
        plt.title('各類別對卡方統計量的貢獻')

        contribution_plot = get_base64_plot()

        return {
            "chi_square_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(df),
            "category_contributions": contributions.tolist(),
            "conclusion": conclusion,
            "plot_base64": plot_base64,
            "contribution_plot": contribution_plot
        }

    except Exception as e:
        raise CalculationError(message=f"卡方檢定計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
