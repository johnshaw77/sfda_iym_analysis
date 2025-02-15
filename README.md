# 統計分析 API 服務

這是一個基於 FastAPI 的統計分析 API 服務，提供多種統計分析方法的 RESTful API 接口。

## 已完成功能
1. 描述性統計分析（Descriptive Statistics）
   - 基本統計量計算（平均數、中位數、標準差等）
   - 分布特性分析（偏度、峰度）
   - 包含直方圖、箱型圖和 Q-Q 圖
   - 提供詳細的描述性結論

2. 相關性分析（Correlation Analysis）
   - 支援 Pearson 相關係數計算
   - 支援 Spearman 等級相關係數
   - 提供相關性視覺化圖表
   - 包含 p 值和置信區間

3. 線性回歸（Linear Regression）
   - 支援簡單線性回歸分析
   - 提供回歸係數和截距
   - 包含 R² 值和模型評估指標
   - 視覺化回歸直線和殘差圖

4. 變異數分析（ANOVA）
   - 支援單因子變異數分析
   - 提供 F 統計量和 p 值
   - 計算效果量（Eta-squared）
   - 包含箱型圖和小提琴圖視覺化
   - 提供詳細的組間比較結果

5. 卡方檢定（Chi-square Test）
   - 支援適合度檢定
   - 提供卡方統計量和 p 值
   - 計算各類別的貢獻值
   - 包含觀察值與期望值比較圖
   - 視覺化卡方貢獻值分布

6. 獨立樣本 t 檢定（Independent t-test）
   - 支援兩組獨立樣本比較
   - 提供 t 統計量和 p 值
   - 計算效果量（Cohen's d）
   - 包含箱型圖比較
   - 提供置信區間

7. 配對樣本 t 檢定（Paired t-test）
   - 支援前後測資料比較
   - 提供 t 統計量和 p 值
   - 計算效果量
   - 包含箱型圖和差異值分布圖
   - 提供詳細的比較結果

8. 存活分析（Survival Analysis）
   - 支援 Kaplan-Meier 估計
   - 支援 Nelson-Aalen 估計
   - 包含存活曲線和累積風險圖
   - 支援分組比較和 log-rank 檢定
   - 提供中位存活時間估計

9. 假設檢定（Hypothesis Test）
   - 支援單樣本 z 檢定
   - 提供檢定統計量和 p 值
   - 包含直方圖和 Q-Q 圖
   - 提供置信區間
   - 詳細的檢定結論

10. 多元迴歸分析（Multiple Regression）
    - 支援多個自變量的迴歸分析
    - 提供各變量的係數和顯著性
    - 包含模型摘要（R²、調整後 R²）
    - 視覺化係數圖和殘差圖
    - 提供預測功能

11. 邏輯迴歸分析（Logistic Regression）
    - 支援二元分類問題
    - 提供各變量的係數和勝算比
    - 包含模型評估指標（準確度、精確度、召回率）
    - ROC 曲線和 AUC 值
    - 提供預測機率

12. 主成分分析（Principal Component Analysis）
    - 支援降維和特徵提取
    - 計算解釋變異比例
    - 提供載荷矩陣和得分
    - 包含碎石圖和累積變異圖
    - 視覺化主成分散點圖

13. 因子分析（Factor Analysis）
    - 支援探索性因子分析
    - 自動決定最佳因子數量（Kaiser 準則）
    - 提供因子載荷和共同性
    - 包含碎石圖和平行分析
    - 支援因子旋轉（Varimax 等）
    - 視覺化因子載荷圖

## 環境設置

### Conda 環境設置
```bash
# 創建 conda 環境
conda create -n py312_stat python=3.12

# 啟動環境
conda activate py312_stat

# 安裝所需套件
pip install -r requirements.txt
```

### 主要依賴套件
- fastapi==0.104.1
- uvicorn==0.24.0
- numpy==1.26.2
- scipy==1.11.4
- pandas==2.1.3
- matplotlib==3.8.2
- seaborn==0.13.0
- scikit-learn==1.3.2
- factor-analyzer==0.5.1
- pydantic-settings==2.7.1
- python-multipart==0.0.6


### 啟動服務
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 功能說明

### 提供的統計分析方法
1. t 檢定
2. 假設檢定
3. 線性回歸
4. 相關性分析
5. 卡方檢定

### API 端點設計
- **POST /analyze**
  - 統一的分析端點，根據方法參數提供不同的統計分析

### 參數設置
用戶可以根據需求傳入以下參數：
- 方法選擇：
  - `descriptive_analysis`：描述性統計分析
  - `correlation_analysis`：相關性分析
  - `linear_regression`：線性回歸
  - `anova`：變異數分析
  - `chi_square_test`：卡方檢定
  - `t_test`：獨立樣本 t 檢定
  - `paired_t_test`：配對樣本 t 檢定
  - `survival_analysis`：存活分析
  - `hypothesis_test`：假設檢定
  - `factor_analysis`：因子分析
- 數據輸入：根據不同方法要求提供相應的數據集

## API 使用範例

### 1. 描述性統計分析
```json
{
  "method": "descriptive_analysis",
  "data": {
    "data": [1.2, 2.3, 3.1, 2.5, 3.8, 4.2]
  }
}
```

### 2. 相關性分析
```json
{
  "method": "correlation_analysis",
  "data": {
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 6, 8, 10]
  }
}
```

### 3. 線性回歸
```json
{
  "method": "linear_regression",
  "data": {
    "x": [1, 2, 3, 4, 5],
    "y": [2.1, 4.2, 5.9, 8.1, 9.8],
    "predict_x": 6
  }
}
```

### 4. 變異數分析
```json
{
  "method": "anova",
  "data": {
    "groups": [
      [1.2, 2.3, 3.1],
      [1.8, 2.5, 3.0],
      [2.1, 2.8, 3.3]
    ],
    "group_names": ["組別1", "組別2", "組別3"]
  }
}
```

### 5. 卡方檢定
```json
{
  "method": "chi_square_test",
  "data": {
    "observed": [10, 20, 30, 40],
    "expected": [15, 25, 35, 25]
  }
}
```

### 6. 獨立樣本 t 檢定
```json
{
  "method": "t_test",
  "data": {
    "group1": [1.2, 2.3, 3.1, 2.8],
    "group2": [1.8, 2.5, 3.0, 3.2]
  }
}
```

### 7. 配對樣本 t 檢定
```json
{
  "method": "paired_t_test",
  "data": {
    "pre_test": [1.2, 2.3, 3.1, 2.8],
    "post_test": [1.8, 2.5, 3.0, 3.2]
  }
}
```

### 8. 存活分析
```json
{
  "method": "survival_analysis",
  "data": {
    "times": [5, 10, 15, 20, 25],
    "events": [1, 1, 0, 1, 0],
    "groups": [1, 1, 2, 2, 2],
    "group_names": ["實驗組", "對照組"]
  }
}
```

### 9. 假設檢定
```json
{
  "method": "hypothesis_test",
  "data": {
    "data": [1.2, 2.3, 3.1, 2.8, 3.5],
    "hypothesis_value": 2.5
  }
}
```

### 10. 多元迴歸分析
```json
{
  "method": "multiple_regression",
  "data": {
    "X": [
      [1, 2, 3, 4, 5],
      [2, 4, 6, 8, 10],
      [1.2, 2.3, 3.1, 2.5, 3.8]
    ],
    "y": [2.1, 4.2, 6.3]
  }
}
```

### 11. 邏輯迴歸分析
```json
{
  "method": "logistic_regression",
  "data": {
    "X": [
      [1, 2, 3, 4, 5],
      [2, 4, 6, 8, 10],
      [1.2, 2.3, 3.1, 2.5, 3.8]
    ],
    "y": [0, 0, 1]
  }
}
```

### 12. 主成分分析
```json
{
  "method": "principal_component_analysis",
  "data": {
    "X": [
      [1.2, 2.3, 3.1, 2.8, 2.5, 3.0],
      [2.1, 3.2, 2.9, 3.1, 2.8, 3.3],
      [1.8, 2.5, 3.0, 2.7, 2.4, 2.9],
      [2.4, 3.1, 2.8, 3.0, 2.6, 3.2]
    ],
    "feature_names": ["智力測驗", "學習動機", "學習態度", "學習成績"],
    "n_components": 2
  }
}
```

### 13. 因子分析
```json
{
  "method": "factor_analysis",
  "data": {
    "X": [
      [1.2, 2.3, 3.1, 2.8, 2.5, 3.0],
      [2.1, 3.2, 2.9, 3.1, 2.8, 3.3],
      [1.8, 2.5, 3.0, 2.7, 2.4, 2.9],
      [2.4, 3.1, 2.8, 3.0, 2.6, 3.2]
    ],
    "feature_names": ["智力測驗", "學習動機", "學習態度", "學習成績"],
    "n_factors": 2,
    "rotation": "varimax"
  }
}
```

## 專案結構
```
sfda_stat/
├── app/
│   ├── main.py              # FastAPI 主程式
│   ├── config.py            # 配置文件
│   ├── models/             
│   │   ├── request.py       # 請求模型
│   │   └── response.py      # 回應模型
│   ├── services/
│   │   ├── t_test.py
│   │   ├── hypothesis_test.py
│   │   ├── linear_regression.py
│   │   ├── correlation.py
│   │   └── chi_square.py
│   ├── utils/
│   │   ├── validators.py    # 數據驗證
│   │   ├── plotting.py      # 圖表生成
│   │   └── errors.py        # 錯誤處理
│   └── tests/
│       ├── test_t_test.py
│       ├── test_hypothesis.py
│       └── ...
├── requirements.txt
└── README.md
```

## 特色功能

### 1. 錯誤處理機制
- 數據格式驗證
- 數據大小限制
- 缺失值處理
- 異常值檢測

### 2. 數據驗證
- 數據長度驗證
- 數值範圍檢查
- 數據類型轉換

### 3. 分析結果
- 主要統計指標
- 置信區間
- 效果量
- 統計檢定力
- 視覺化圖表（base64 或 URL）

### 4. 安全性考慮
- 請求速率限制
- 數據大小限制
- API 認證機制

## 開發規劃

### 第一階段
1. 建立基本專案結構
2. 實現數據驗證邏輯
3. 設置開發環境

### 第二階段
1. 實現核心統計方法
2. 加入單元測試
3. 整合錯誤處理

### 第三階段
1. 實現視覺化功能
2. 優化效能
3. 加強安全性

### 第四階段
1. API 文檔完善
2. 部署準備
3. 監控整合

## 維護與支援
- 問題回報
- 功能建議
- 版本更新 