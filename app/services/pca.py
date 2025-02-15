from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError, ValidationError


def perform_pca(X: List[List[float]],
                feature_names: Optional[List[str]] = None,
                n_components: Optional[int] = None) -> Dict[str, Any]:
    """
    執行主成分分析
    
    參數:
        X: 輸入數據矩陣，每個內部列表代表一個特徵的所有觀測值
        feature_names: 特徵名稱列表（可選）
        n_components: 要保留的主成分數量（可選，默認為None，表示保留所有成分）
        
    返回:
        包含主成分分析結果的字典
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

        # 執行 PCA
        if n_components is None:
            n_components = min(n_samples, n_features)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # 計算解釋變異比例
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # 計算貢獻度
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # 生成碎石圖
        plt.figure(figsize=(10, 6))
        plt.plot(range(1,
                       len(explained_variance_ratio) + 1),
                 explained_variance_ratio, 'bo-')
        plt.xlabel('主成分')
        plt.ylabel('解釋變異比例')
        plt.title('碎石圖')
        scree_plot = get_base64_plot()

        # 生成累積變異圖
        plt.figure(figsize=(10, 6))
        plt.plot(range(1,
                       len(cumulative_variance_ratio) + 1),
                 cumulative_variance_ratio, 'ro-')
        plt.axhline(y=0.8, color='g', linestyle='--', label='80% 閾值')
        plt.xlabel('主成分數量')
        plt.ylabel('累積解釋變異比例')
        plt.title('累積變異圖')
        plt.legend()
        cumulative_plot = get_base64_plot()

        # 如果有兩個以上的主成分，生成二維散點圖
        scatter_plot = None
        if n_components >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1])
            plt.xlabel('第一主成分')
            plt.ylabel('第二主成分')
            plt.title('前兩個主成分的散點圖')
            scatter_plot = get_base64_plot()

        # 生成載荷圖（前兩個主成分的特徵載荷）
        loading_plot = None
        if n_components >= 2:
            plt.figure(figsize=(10, 6))
            loading_matrix = pca.components_[:2].T
            plt.scatter(loading_matrix[:, 0], loading_matrix[:, 1])
            for i, feature in enumerate(feature_names):
                plt.annotate(feature,
                             (loading_matrix[i, 0], loading_matrix[i, 1]))
            plt.xlabel('第一主成分載荷')
            plt.ylabel('第二主成分載荷')
            plt.title('特徵載荷圖')
            # 添加單位圓
            circle = plt.Circle((0, 0),
                                1,
                                fill=False,
                                linestyle='--',
                                color='gray')
            plt.gca().add_artist(circle)
            plt.axis('equal')
            loading_plot = get_base64_plot()

        # 生成熱圖
        plt.figure(figsize=(10, 8))
        sns.heatmap(loadings,
                    xticklabels=[f'PC{i+1}' for i in range(n_components)],
                    yticklabels=feature_names,
                    cmap='RdBu',
                    center=0,
                    annot=True,
                    fmt='.2f')
        plt.title('特徵載荷熱圖')
        heatmap_plot = get_base64_plot()

        # 準備主成分結果
        pc_results = []
        for i in range(n_components):
            pc_result = {
                "component_number": i + 1,
                "explained_variance_ratio": float(explained_variance_ratio[i]),
                "cumulative_variance_ratio":
                float(cumulative_variance_ratio[i]),
                "loadings": {
                    feature_names[j]: float(loadings[j, i])
                    for j in range(n_features)
                }
            }
            pc_results.append(pc_result)

        # 生成結論
        conclusion = f"分析共提取了 {n_components} 個主成分。\n"
        conclusion += f"第一主成分解釋了 {explained_variance_ratio[0]:.2%} 的總變異。\n"
        if n_components >= 2:
            conclusion += f"前兩個主成分共解釋了 {cumulative_variance_ratio[1]:.2%} 的總變異。\n"

        # 找出對每個主成分貢獻最大的特徵
        for i in range(min(3, n_components)):  # 只報告前三個主成分
            pc_loadings = loadings[:, i]
            max_loading_idx = np.abs(pc_loadings).argmax()
            conclusion += f"第 {i+1} 主成分主要受 {feature_names[max_loading_idx]} 的影響"
            conclusion += f"（載荷值：{pc_loadings[max_loading_idx]:.3f}）。\n"

        # 建議保留的主成分數量
        n_suggested = np.sum(cumulative_variance_ratio <= 0.8) + 1
        conclusion += f"\n建議保留 {n_suggested} 個主成分，可以解釋 {cumulative_variance_ratio[n_suggested-1]:.2%} 的總變異。"

        return {
            "summary": {
                "n_components": n_components,
                "n_features": n_features,
                "n_samples": n_samples,
                "total_variance_explained":
                float(cumulative_variance_ratio[-1])
            },
            "principal_components": pc_results,
            "transformed_data": X_pca.tolist(),
            "feature_names": feature_names,
            "plots": {
                "scree_plot": scree_plot,
                "cumulative_plot": cumulative_plot,
                "scatter_plot": scatter_plot,
                "loading_plot": loading_plot,
                "heatmap_plot": heatmap_plot
            },
            "conclusion": conclusion
        }

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"主成分分析過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
