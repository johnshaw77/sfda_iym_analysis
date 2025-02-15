from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array)
from ..utils.plotting import get_base64_plot, get_chinese_font
from ..utils.errors import CalculationError, ValidationError


def perform_anova(groups: List[List[float]],
                  group_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    執行單因子變異數分析 (One-way ANOVA)
    
    參數:
        groups: 多組數據的列表，例如 [[1.2, 2.3, 3.1], [1.8, 2.5, 3.0], [2.1, 2.8, 3.3]]
        group_names: 組別名稱列表（可選），例如 ["組別1", "組別2", "組別3"]
        
    返回:
        包含分析結果的字典
    """
    try:
        # 數據驗證
        if not groups:
            raise ValidationError("數據不能為空")

        if len(groups) < 2:
            raise ValidationError("ANOVA 分析需要至少兩組數據")

        # 驗證每組數據
        for i, group in enumerate(groups):
            if not group:
                raise ValidationError(f"第 {i+1} 組數據不能為空")
            validate_array_size(group, f"group_{i+1}")
            validate_sample_size(group, f"group_{i+1}")
            validate_numeric_array(group, f"group_{i+1}")

        # 轉換為 numpy 數組並檢查有效性
        groups_arr = [np.array(group, dtype=float) for group in groups]
        for i, group in enumerate(groups_arr):
            if np.any(np.isnan(group)) or np.any(np.isinf(group)):
                raise ValidationError(f"第 {i+1} 組數據包含無效值（NaN 或無限值）")

        # 設置組別名稱
        if group_names is None or len(group_names) != len(groups):
            group_names = [f"組別{i+1}" for i in range(len(groups))]

        # 執行 ANOVA
        f_stat, p_value = stats.f_oneway(*groups_arr)

        # 計算各組的描述性統計
        group_stats = []
        for i, group in enumerate(groups_arr):
            stats_dict = {
                "name": group_names[i],
                "size": len(group),
                "mean": float(np.mean(group)),
                "std": float(np.std(group, ddof=1)),
                "min": float(np.min(group)),
                "max": float(np.max(group))
            }
            group_stats.append(stats_dict)

        # 計算效果量 (Eta-squared)
        all_data = np.concatenate(groups_arr)
        grand_mean = np.mean(all_data)
        ss_between = sum(
            len(group) * (np.mean(group) - grand_mean)**2
            for group in groups_arr)
        ss_total = sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0

        # 生成結論
        conclusion = "拒絕虛無假設" if p_value < 0.05 else "無法拒絕虛無假設"
        if p_value < 0.05:
            conclusion += "，各組間存在顯著差異"
            if eta_squared > 0.14:
                conclusion += "，效果量大"
            elif eta_squared > 0.06:
                conclusion += "，效果量中等"
            else:
                conclusion += "，效果量小"
        else:
            conclusion += "，未發現顯著差異"

        # 設置 matplotlib 樣式
        plt.style.use('default')
        sns.set_style("whitegrid")

        # 設置中文字體
        font = get_chinese_font()

        # 生成箱型圖
        plt.figure(figsize=(10, 6))
        box_data = [group for group in groups_arr]
        sns.boxplot(data=box_data)
        if font:
            plt.xticks(range(len(group_names)),
                       group_names,
                       fontproperties=font)
            plt.title('各組數據分布箱型圖', fontproperties=font)
            plt.xlabel('組別', fontproperties=font)
            plt.ylabel('數值', fontproperties=font)
        else:
            plt.xticks(range(len(group_names)), group_names)
            plt.title('各組數據分布箱型圖')
            plt.xlabel('組別')
            plt.ylabel('數值')
        box_plot = get_base64_plot()

        # 生成小提琴圖
        plt.figure(figsize=(10, 6))
        violin_data = [group for group in groups_arr]
        sns.violinplot(data=violin_data)
        if font:
            plt.xticks(range(len(group_names)),
                       group_names,
                       fontproperties=font)
            plt.title('各組數據分布小提琴圖', fontproperties=font)
            plt.xlabel('組別', fontproperties=font)
            plt.ylabel('數值', fontproperties=font)
        else:
            plt.xticks(range(len(group_names)), group_names)
            plt.title('各組數據分布小提琴圖')
            plt.xlabel('組別')
            plt.ylabel('數值')
        violin_plot = get_base64_plot()

        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "eta_squared": float(eta_squared),
            "group_statistics": group_stats,
            "conclusion": conclusion,
            "box_plot": box_plot,
            "violin_plot": violin_plot
        }

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"ANOVA 分析計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
