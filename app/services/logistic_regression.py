from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import get_base64_plot
from ..utils.errors import CalculationError, ValidationError


def perform_logistic_regression(
        X: List[List[float]],
        y: List[int],
        feature_names: Optional[List[str]] = None,
        predict_X: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    執行邏輯迴歸分析
    
    參數:
        X: 自變量矩陣，每個內部列表代表一個特徵的所有觀測值
        y: 二元因變量數據（0 或 1）
        feature_names: 特徵名稱列表（可選）
        predict_X: 用於預測的新數據點（可選）
        
    返回:
        包含邏輯迴歸分析結果的字典
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
        if not all(
                isinstance(val, (int, np.integer)) and val in [0, 1]
                for val in y):
            raise ValidationError("因變量必須為二元數據（0 或 1）")
        if len(y) != n_samples:
            raise ValidationError("因變量的樣本數量必須與特徵相同")

        # 轉換數據格式
        X_array = np.array(X).T  # 轉置為樣本 × 特徵的格式
        y_array = np.array(y)

        # 標準化特徵
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # 建立並擬合模型
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y_array)

        # 計算預測值和機率
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # 計算模型評估指標
        accuracy = accuracy_score(y_array, y_pred)
        precision = precision_score(y_array, y_pred)
        recall = recall_score(y_array, y_pred)
        f1 = f1_score(y_array, y_pred)

        # 計算 Pseudo R² (McFadden's R²)
        null_model = LogisticRegression(fit_intercept=True, random_state=42)
        null_model.fit(np.ones((X_scaled.shape[0], 1)), y_array)
        ll_null = null_model.score(np.ones((X_scaled.shape[0], 1)), y_array)
        ll_model = model.score(X_scaled, y_array)
        pseudo_r2 = 1 - (ll_model / ll_null)

        # 計算各係數的 Wald 檢定
        coef_summary = []
        for i in range(n_features):
            name = feature_names[i] if feature_names else f"特徵{i+1}"
            coef = model.coef_[0][i]
            odds_ratio = np.exp(coef)

            # 計算標準誤和 z 值
            probs = model.predict_proba(X_scaled)[:, 1]
            W = np.diag(probs * (1 - probs))
            X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
            var_covar = np.linalg.inv(X_design.T @ W @ X_design)
            std_err = np.sqrt(var_covar[i + 1, i + 1])
            z_value = coef / std_err
            p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))

            coef_summary.append({
                "name": name,
                "coefficient": float(coef),
                "odds_ratio": float(odds_ratio),
                "std_error": float(std_err),
                "z_value": float(z_value),
                "p_value": float(p_value)
            })

        # 生成預測值（如果提供了新數據）
        prediction = None
        prediction_prob = None
        if predict_X is not None:
            if len(predict_X) != n_features:
                raise ValidationError("預測數據的特徵數量必須與訓練數據相同")
            X_pred = np.array(predict_X).reshape(1, -1)
            X_pred_scaled = scaler.transform(X_pred)
            prediction = int(model.predict(X_pred_scaled)[0])
            prediction_prob = float(model.predict_proba(X_pred_scaled)[0, 1])

        # 生成 ROC 曲線
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_array, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=2,
                 label=f'ROC 曲線 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('偽陽性率')
        plt.ylabel('真陽性率')
        plt.title('接收者操作特徵曲線 (ROC)')
        plt.legend(loc="lower right")
        roc_plot = get_base64_plot()

        # 生成係數圖
        plt.figure(figsize=(10, 6))
        coef_names = [c["name"] for c in coef_summary]
        coef_values = [c["coefficient"] for c in coef_summary]
        plt.barh(range(len(coef_values)), coef_values)
        plt.yticks(range(len(coef_names)), coef_names)
        plt.xlabel('係數值')
        plt.title('邏輯迴歸係數')
        coefficient_plot = get_base64_plot()

        # 生成預測概率分布圖
        plt.figure(figsize=(8, 6))
        sns.histplot(data=y_prob, bins=30, stat='density')
        plt.xlabel('預測機率')
        plt.ylabel('密度')
        plt.title('預測機率分布')
        probability_plot = get_base64_plot()

        # 生成結論
        conclusion = f"模型的準確度為 {accuracy:.3f}，精確度為 {precision:.3f}，"
        conclusion += f"召回率為 {recall:.3f}，F1 分數為 {f1:.3f}。\n"
        conclusion += f"McFadden's Pseudo R² 為 {pseudo_r2:.3f}，"
        conclusion += f"ROC 曲線下面積 (AUC) 為 {roc_auc:.3f}。\n"

        significant_features = [
            c["name"] for c in coef_summary if c["p_value"] < 0.05
        ]
        if significant_features:
            conclusion += f"顯著的特徵包括：{', '.join(significant_features)}。"

        return {
            "model_summary": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "pseudo_r2": float(pseudo_r2),
                "auc_roc": float(roc_auc),
                "sample_size": n_samples,
                "feature_count": n_features
            },
            "coefficients": coef_summary,
            "intercept": float(model.intercept_[0]),
            "prediction": prediction,
            "prediction_probability": prediction_prob,
            "plots": {
                "roc_plot": roc_plot,
                "coefficient_plot": coefficient_plot,
                "probability_plot": probability_plot
            },
            "conclusion": conclusion
        }

    except ValidationError as e:
        raise e
    except Exception as e:
        raise CalculationError(message=f"邏輯迴歸分析過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
