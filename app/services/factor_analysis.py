from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError, ValidationError


def perform_factor_analysis(X: List[List[float]],
                            feature_names: Optional[List[str]] = None,
                            n_factors: Optional[int] = None,
                            rotation: str = 'varimax') -> Dict[str, Any]:
    """
    執行因子分析
    
    參數:
        X: 輸入數據矩陣，每個內部列表代表一個特徵的所有觀測值
        feature_names: 特徵名稱列表（可選）
        n_factors: 要提取的因子數量（可選）
        rotation: 旋轉方法，默認為 'varimax'
        
    返回:
        包含因子分析結果的字典
    """
    try:
        # 數據驗證
        if not X:
            raise ValidationError("數據不能為空")

        n_features = len(X)
        n_samples = len(X[0])

        # 驗證每個特徵的數據
        for i, feature in enumerate(X):
            validate_array_size(feature, f"feature_{i+1}")
            validate_numeric_array(feature, f"feature_{i+1}")
            if len(feature) != n_samples:
                raise ValidationError("所有特徵的樣本數量必須相同")

        # 設置特徵名稱
        if feature_names is None:
            feature_names = [f"特徵{i+1}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValidationError("特徵名稱的數量必須與特徵數量相同")

        # 轉換數據格式並標準化
        X_array = np.array(X).T  # 轉置為樣本 × 特徵的格式
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # 計算 KMO 和 Bartlett 球形度檢定，並進行平行分析
        fa_parallel = FactorAnalyzer(rotation=None, n_factors=n_features)
        fa_parallel.fit(X_scaled)
        ev, ev_parallel = fa_parallel.get_eigenvalues()

        # 如果未指定因子數量，使用 Kaiser 準則
        if n_factors is None:
            n_factors = sum(ev > 1)  # Kaiser 準則
            if n_factors == 0:  # 如果沒有特徵值大於1
                n_factors = 1

        # 執行因子分析
        fa = FactorAnalyzer(rotation=rotation, n_factors=n_factors)
        fa.fit(X_scaled)

        # 獲取因子載荷
        loadings = fa.loadings_
        variance = fa.get_factor_variance()
        communalities = fa.get_communalities()

        # 生成碎石圖
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(ev) + 1), ev, 'bo-', label='實際特徵值')
        plt.plot(range(1,
                       len(ev_parallel) + 1),
                 ev_parallel,
                 'ro--',
                 label='平行分析')
        plt.axhline(y=1, color='g', linestyle='--', label='Kaiser 準則')
        plt.xlabel('因子數量')
        plt.ylabel('特徵值')
        plt.title('碎石圖與平行分析')
        plt.legend()
        scree_plot = get_base64_plot()

        # 生成因子載荷熱圖
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings,
                    xticklabels=[f'因子{i+1}' for i in range(n_factors)],
                    yticklabels=feature_names,
                    cmap='RdBu',
                    center=0,
                    annot=True,
                    fmt='.2f')
        plt.title('因子載荷熱圖')
        loading_plot = get_base64_plot()

        # 如果有兩個以上的因子，生成因子載荷散點圖
        loading_scatter = None
        if n_factors >= 2:
            plt.figure(figsize=(10, 10))
            plt.scatter(loadings[:, 0], loadings[:, 1])
            for i, feature in enumerate(feature_names):
                plt.annotate(feature, (loadings[i, 0], loadings[i, 1]))
            plt.xlabel('第一因子')
            plt.ylabel('第二因子')
            plt.title('因子載荷散點圖')
            plt.axhline(y=0, color='k', linestyle='--')
            plt.axvline(x=0, color='k', linestyle='--')
            loading_scatter = get_base64_plot()

        # 準備因子結果
        factor_results = []
        for i in range(n_factors):
            factor_result = {
                "factor_number": i + 1,
                "variance_explained": float(variance[0][i]),
                "variance_ratio": float(variance[1][i]),
                "cumulative_variance": float(variance[2][i]),
                "loadings": {
                    feature_names[j]: float(loadings[j, i])
                    for j in range(n_features)
                }
            }
            factor_results.append(factor_result)

        # 生成結論
        conclusion = f"分析共提取了 {n_factors} 個因子。\n"
        conclusion += f"第一個因子解釋了 {variance[1][0]:.2%} 的總變異。\n"
        if n_factors >= 2:
            conclusion += f"前兩個因子共解釋了 {variance[2][1]:.2%} 的總變異。\n"

        # 找出每個因子的主要特徵
        for i in range(n_factors):
            significant_features = [
                feature_names[j] for j in range(n_features)
                if abs(loadings[j, i]) > 0.4
            ]
            if significant_features:
                conclusion += f"\n因子 {i+1} 主要與以下特徵相關：{', '.join(significant_features)}。"

        return {
            "summary": {
                "n_factors": n_factors,
                "n_features": n_features,
                "n_samples": n_samples,
                "total_variance_explained": float(variance[2][-1]),
                "rotation_method": rotation
            },
            "factors": factor_results,
            "communalities": {
                feature_names[i]: float(communalities[i])
                for i in range(n_features)
            },
            "feature_names": feature_names,
            "plots": {
                "scree_plot": scree_plot,
                "loading_plot": loading_plot,
                "loading_scatter": loading_scatter
            },
            "conclusion": conclusion
        }

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"因子分析過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
