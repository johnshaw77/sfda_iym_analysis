from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError, ValidationError


def perform_survival_analysis(
        times: List[float],
        events: List[int],
        groups: Optional[List[int]] = None,
        group_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    執行存活分析
    
    參數:
        times: 時間數據列表
        events: 事件指標列表（1表示事件發生，0表示審查）
        groups: 分組指標列表（可選）
        group_names: 組別名稱列表（可選）
        
    返回:
        包含分析結果的字典
    """
    try:
        # 數據驗證
        if not times or not events:
            raise ValidationError("數據不能為空")

        validate_array_size(times, "times")
        validate_array_size(events, "events")
        validate_numeric_array(times, "times")
        validate_numeric_array(events, "events")
        validate_equal_length(times, events, "times", "events")

        # 檢查事件指標是否只包含 0 和 1
        events_arr = np.array(events)
        if not np.all(np.isin(events_arr, [0, 1])):
            raise ValidationError("事件指標只能包含 0（審查）和 1（事件發生）")

        # 轉換為 numpy 數組
        times_arr = np.array(times, dtype=float)

        # 檢查是否包含 NaN 或無限值
        if np.any(np.isnan(times_arr)) or np.any(np.isinf(times_arr)):
            raise ValidationError("時間數據包含無效值（NaN 或無限值）")

        # 檢查時間是否都為正數
        if np.any(times_arr <= 0):
            raise ValidationError("時間數據必須為正數")

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

        # 初始化結果字典
        result = {}

        if groups is not None and group_names is not None:
            # 驗證分組數據
            validate_equal_length(times, groups, "times", "groups")
            unique_groups = np.unique(groups)
            if len(unique_groups) != len(group_names):
                raise ValidationError("組別數量與組別名稱數量不匹配")

            # 執行分組存活分析
            plt.figure(figsize=(10, 6))
            kmfs = []
            nafs = []
            for i, group in enumerate(unique_groups):
                mask = np.array(groups) == group
                # KM 估計
                kmf = KaplanMeierFitter()
                kmf.fit(times_arr[mask],
                        events_arr[mask],
                        label=group_names[i])
                kmfs.append(kmf)
                kmf.plot()

                # Nelson-Aalen 估計
                naf = NelsonAalenFitter()
                naf.fit(times_arr[mask],
                        events_arr[mask],
                        label=group_names[i])
                nafs.append(naf)

            plt.title('分組存活曲線')
            plt.xlabel('時間')
            plt.ylabel('存活率')
            survival_plot = get_base64_plot()

            # 執行 log-rank 檢定
            if len(unique_groups) == 2:
                group1_mask = np.array(groups) == unique_groups[0]
                group2_mask = np.array(groups) == unique_groups[1]
                log_rank_result = logrank_test(times_arr[group1_mask],
                                               times_arr[group2_mask],
                                               events_arr[group1_mask],
                                               events_arr[group2_mask])
                result["log_rank_test"] = {
                    "statistic": float(log_rank_result.test_statistic),
                    "p_value": float(log_rank_result.p_value)
                }

        else:
            # 執行整體存活分析
            kmf = KaplanMeierFitter()
            naf = NelsonAalenFitter()

            # KM 估計
            kmf.fit(times_arr, events_arr, label='整體')
            # Nelson-Aalen 估計
            naf.fit(times_arr, events_arr, label='整體')

            # 生成存活曲線圖
            plt.figure(figsize=(10, 6))
            kmf.plot()
            plt.title('存活曲線')
            plt.xlabel('時間')
            plt.ylabel('存活率')
            survival_plot = get_base64_plot()

            # 計算中位存活時間
            median_survival = float(kmf.median_survival_time_)
            result["median_survival"] = median_survival

            # 計算存活率在特定時間點的值
            time_points = [np.percentile(times_arr, p) for p in [25, 50, 75]]
            survival_probs = {
                f"at_{p}": float(kmf.survival_function_at_times(t).iloc[0])
                for p, t in zip(['q1', 'median', 'q3'], time_points)
            }
            result["survival_probabilities"] = survival_probs

        # 生成累積風險圖
        plt.figure(figsize=(10, 6))
        if groups is not None and group_names is not None:
            for naf in nafs:
                naf.plot()
        else:
            naf.plot()
        plt.title('累積風險圖')
        plt.xlabel('時間')
        plt.ylabel('累積風險')
        cumulative_plot = get_base64_plot()

        # 生成基本統計信息
        total_subjects = len(times)
        total_events = sum(events)
        censored = total_subjects - total_events

        # 生成結論
        if groups is not None and group_names is not None:
            conclusion = (
                f"分析包含 {total_subjects} 個受試者，分為 {len(unique_groups)} 組。"
                f"總共發生 {total_events} 個事件，{censored} 個被審查。")
            if "log_rank_test" in result:
                p_value = result["log_rank_test"]["p_value"]
                conclusion += f"\n組間存活曲線" + ("有顯著差異"
                                             if p_value < 0.05 else "無顯著差異")
                conclusion += f"（log-rank 檢定 p 值 = {p_value:.4f}）"
        else:
            conclusion = (f"分析包含 {total_subjects} 個受試者，"
                          f"其中 {total_events} 個事件發生，"
                          f"{censored} 個被審查。")
            if "median_survival" in result:
                conclusion += f"\n中位存活時間為 {result['median_survival']:.2f}。"

        # 更新結果字典
        result.update({
            "sample_info": {
                "total_subjects": total_subjects,
                "total_events": total_events,
                "censored": censored
            },
            "conclusion": conclusion,
            "survival_plot": survival_plot,
            "cumulative_plot": cumulative_plot
        })

        return result

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"存活分析過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
