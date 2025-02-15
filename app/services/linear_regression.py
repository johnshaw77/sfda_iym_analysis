from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from ..utils.validators import (validate_array_size, validate_sample_size,
                                validate_numeric_array, validate_equal_length)
from ..utils.plotting import create_regression_plot, create_residual_plot
from ..utils.errors import CalculationError


def perform_linear_regression(
        x: List[float],
        y: List[float],
        predict_x: Optional[float] = None) -> Dict[str, Any]:
    """
    執行線性回歸分析
    
    參數:
        x: 自變量數據
        y: 因變量數據
        predict_x: 預測值的 x 值（可選）
        
    返回:
        包含回歸分析結果的字典
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
        # 將數據轉換為 numpy 數組並重塑
        X = np.array(x).reshape(-1, 1)
        Y = np.array(y)

        # 建立並擬合模型
        model = LinearRegression()
        model.fit(X, Y)

        # 獲取係數和截距
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        # 計算預測值
        y_pred = model.predict(X)

        # 計算 R² 值
        r_squared = r2_score(Y, y_pred)

        # 計算標準誤差
        n = len(x)
        standard_error = np.sqrt(
            mean_squared_error(Y, y_pred) * (n - 1) / (n - 2))

        # 計算 p 值
        x_mean = np.mean(x)
        x_squared_sum = np.sum((np.array(x) - x_mean)**2)
        t_stat = slope / (standard_error / np.sqrt(x_squared_sum))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        # 生成回歸方程
        equation = f"y = {slope:.2f}x + {intercept:.2f}"

        # 如果提供了預測值，計算預測結果
        predicted_y = None
        if predict_x is not None:
            predicted_y = float(model.predict([[predict_x]])[0])

        # 生成圖表
        plot_base64 = create_regression_plot(x=x,
                                             y=y,
                                             coef=slope,
                                             intercept=intercept,
                                             title="線性回歸分析",
                                             xlabel="X",
                                             ylabel="Y")

        # 計算殘差
        residuals = Y - y_pred
        residual_plot = create_residual_plot(x=y_pred.tolist(),
                                             residuals=residuals.tolist(),
                                             title="殘差圖",
                                             xlabel="預測值",
                                             ylabel="殘差")

        return {
            "coefficients": [float(intercept), float(slope)],
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "standard_error": float(standard_error),
            "equation": equation,
            "predicted_y": predicted_y,
            "plot_base64": plot_base64,
            "residual_plot": residual_plot
        }

    except Exception as e:
        raise CalculationError(message=f"線性回歸計算過程中發生錯誤: {str(e)}",
                               params={"error": str(e)})
