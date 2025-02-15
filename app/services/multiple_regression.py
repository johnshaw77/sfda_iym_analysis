from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError, ValidationError


def perform_multiple_regression(
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
        predict_X: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    執行多元迴歸分析
    
    參數:
        X: 自變量矩陣，每個內部列表代表一個特徵的所有觀測值
        y: 因變量數據
        feature_names: 特徵名稱列表（可選）
        predict_X: 用於預測的新數據點（可選）
        
    返回:
        包含多元迴歸分析結果的字典
    """
    try:
        # 數據驗證
        if not X or not y:
            raise ValidationError("數據不能為空")

        n_features = len(X)
        n_samples = len(X[0])

        # 驗證每個特徵的數據
        for i, feature in enumerate(X):
            validate_array_size(feature, f"feature_{i+1}")
            validate_numeric_array(feature, f"feature_{i+1}")
            if len(feature) != n_samples:
                raise ValidationError("所有特徵的樣本數量必須相同")

        validate_array_size(y, "y")
        validate_numeric_array(y, "y")
        if len(y) != n_samples:
            raise ValidationError("因變量的樣本數量必須與特徵相同")

        # 轉換數據格式
        X_array = np.array(X).T  # 轉置為樣本 × 特徵的格式
        y_array = np.array(y)

        # 標準化特徵
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # 建立並擬合模型
        model = LinearRegression()
        model.fit(X_scaled, y_array)

        # 計算預測值和殘差
        y_pred = model.predict(X_scaled)
        residuals = y_array - y_pred

        # 計算模型評估指標
        r2 = r2_score(y_array, y_pred)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        mse = mean_squared_error(y_array, y_pred)
        rmse = np.sqrt(mse)

        # 計算 F 檢定統計量和 p 值
        f_stat, f_pvalue = f_regression(X_scaled, y_array)

        # 計算各係數的 t 檢定和 p 值
        n = X_scaled.shape[0]
        p = X_scaled.shape[1]
        dof = n - p - 1
        mse = np.sum(residuals**2) / dof
        var_b = mse * np.linalg.inv(np.dot(X_scaled.T, X_scaled)).diagonal()
        sd_b = np.sqrt(var_b)
        t_stat = model.coef_ / sd_b
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))

        # 生成係數摘要
        coef_summary = []
        for i in range(n_features):
            name = feature_names[i] if feature_names else f"特徵{i+1}"
            coef_summary.append({
                "name": name,
                "coefficient": float(model.coef_[i]),
                "std_error": float(sd_b[i]),
                "t_statistic": float(t_stat[i]),
                "p_value": float(p_values[i])
            })

        # 生成預測值（如果提供了新數據）
        prediction = None
        if predict_X is not None:
            if len(predict_X) != n_features:
                raise ValidationError("預測數據的特徵數量必須與訓練數據相同")
            X_pred = np.array(predict_X).reshape(1, -1)
            X_pred_scaled = scaler.transform(X_pred)
            prediction = float(model.predict(X_pred_scaled)[0])

        # 生成殘差圖
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('預測值')
        plt.ylabel('殘差')
        plt.title('殘差圖')
        residual_plot = get_base64_plot()

        # 生成實際值vs預測值圖
        plt.figure(figsize=(10, 6))
        plt.scatter(y_array, y_pred)
        plt.plot([y_array.min(), y_array.max()],
                 [y_array.min(), y_array.max()],
                 'r--',
                 lw=2)
        plt.xlabel('實際值')
        plt.ylabel('預測值')
        plt.title('實際值 vs 預測值')
        prediction_plot = get_base64_plot()

        # 生成係數圖
        plt.figure(figsize=(12, 6))
        coef_names = [c["name"] for c in coef_summary]
        coef_values = [c["coefficient"] for c in coef_summary]
        plt.barh(range(len(coef_values)), coef_values)
        plt.yticks(range(len(coef_names)), coef_names)
        plt.xlabel('係數值')
        plt.title('迴歸係數')
        coefficient_plot = get_base64_plot()

        # 生成結論
        conclusion = f"模型的判定係數(R²)為 {r2:.3f}，調整後的 R² 為 {adj_r2:.3f}。"
        if r2 > 0.7:
            conclusion += "\n模型具有較好的解釋力。"
        elif r2 > 0.5:
            conclusion += "\n模型具有中等的解釋力。"
        else:
            conclusion += "\n模型的解釋力較弱。"

        significant_features = [
            c["name"] for c in coef_summary if c["p_value"] < 0.05
        ]
        if significant_features:
            conclusion += f"\n顯著的特徵包括：{', '.join(significant_features)}。"

        return {
            "model_summary": {
                "r_squared": float(r2),
                "adjusted_r_squared": float(adj_r2),
                "rmse": float(rmse),
                "f_statistic": float(np.mean(f_stat)),
                "f_p_value": float(np.mean(f_pvalue)),
                "sample_size": n_samples,
                "feature_count": n_features
            },
            "coefficients": coef_summary,
            "intercept": float(model.intercept_),
            "prediction": prediction,
            "plots": {
                "residual_plot": residual_plot,
                "prediction_plot": prediction_plot,
                "coefficient_plot": coefficient_plot
            },
            "conclusion": conclusion
        }

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"多元迴歸分析過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
