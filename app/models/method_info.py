from pydantic import BaseModel
from typing import List, Dict, Optional


class StatisticalMethodInfo(BaseModel):
    """統計方法資訊模型"""
    method_id: str
    name: str
    description: str
    use_cases: List[str]
    required_data: Dict[str, str]
    assumptions: List[str]
    output_metrics: List[str]
    visualization: List[str]
    example: Dict[str, object]


def get_available_methods() -> List[StatisticalMethodInfo]:
    """取得所有可用的統計方法資訊"""
    return [
        StatisticalMethodInfo(
            method_id="descriptive",
            name="描述性統計分析",
            description="對數據進行基本的統計描述，包含集中趨勢和離散趨勢的測量",
            use_cases=["了解數據的基本特徵", "檢查數據的分布情況", "識別可能的異常值", "為進一步的統計分析做準備"],
            required_data={"data": "單一變量的數值型數據"},
            assumptions=["數據為數值型", "數據應為有效值（非缺失值）"],
            output_metrics=[
                "平均數、中位數、眾數", "標準差、變異數", "偏度、峰度", "四分位數", "最大值、最小值"
            ],
            visualization=["直方圖", "箱型圖", "Q-Q圖"],
            example={"data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2]}),
        StatisticalMethodInfo(
            method_id="t_test",
            name="獨立樣本 t 檢定",
            description="比較兩個獨立組別的平均數差異是否具有統計顯著性",
            use_cases=["比較兩種處理方法的效果", "比較兩個群體的表現差異", "評估實驗組和對照組的差異"],
            required_data={
                "group1": "第一組數值型數據",
                "group2": "第二組數值型數據"
            },
            assumptions=["數據為數值型", "兩組數據相互獨立", "數據近似常態分布", "兩組數據具有相近的變異數"],
            output_metrics=["t 統計量", "p 值", "效果量（Cohen's d）", "置信區間"],
            visualization=["箱型圖", "小提琴圖"],
            example={
                "group1": [1.2, 2.3, 3.1, 2.8],
                "group2": [1.8, 2.5, 3.0, 3.2]
            }),
        StatisticalMethodInfo(
            method_id="paired_t_test",
            name="配對樣本 t 檢定",
            description="比較相依樣本在兩種條件下的平均數差異",
            use_cases=["前後測比較", "配對實驗設計", "重複測量數據分析"],
            required_data={
                "pre_test": "前測數值型數據",
                "post_test": "後測數值型數據"
            },
            assumptions=["數據為數值型", "前後測配對關係明確", "差異分數近似常態分布"],
            output_metrics=["t 統計量", "p 值", "效果量", "置信區間"],
            visualization=["配對箱型圖", "差異值分布圖"],
            example={
                "pre_test": [1.2, 2.3, 3.1, 2.8],
                "post_test": [1.8, 2.5, 3.0, 3.2]
            }),
        StatisticalMethodInfo(
            method_id="correlation",
            name="相關性分析",
            description="測量兩個變量之間的線性關係強度",
            use_cases=["探索變量間的關聯性", "識別潛在的預測變量", "評估測量工具的效度"],
            required_data={
                "x": "第一個變量的數值型數據",
                "y": "第二個變量的數值型數據"
            },
            assumptions=["數據為數值型", "變量間具有線性關係", "無極端異常值"],
            output_metrics=["相關係數（Pearson's r）", "p 值", "置信區間"],
            visualization=["散點圖", "相關矩陣熱圖"],
            example={
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10]
            }),
        StatisticalMethodInfo(
            method_id="linear_regression",
            name="線性回歸分析",
            description="建立預測模型以了解自變量對應變量的影響",
            use_cases=["預測數值型結果", "了解變量間的因果關係", "評估預測變量的重要性"],
            required_data={
                "x": "自變量數值型數據",
                "y": "應變量數值型數據"
            },
            assumptions=["變量間具有線性關係", "殘差呈常態分布", "殘差具有同質性", "觀察值獨立"],
            output_metrics=["迴歸係數", "R平方值", "調整後R平方", "p 值", "標準誤"],
            visualization=["散點圖與迴歸線", "殘差圖", "Q-Q圖"],
            example={
                "x": [1, 2, 3, 4, 5],
                "y": [2.1, 4.2, 5.9, 8.1, 9.8]
            }),
        StatisticalMethodInfo(
            method_id="multiple_regression",
            name="多元迴歸分析",
            description="使用多個自變量預測一個應變量",
            use_cases=["複雜預測模型建立", "控制混淆變量", "評估多個預測變量的相對重要性"],
            required_data={
                "X": "多個自變量的數值型數據矩陣",
                "y": "應變量數值型數據"
            },
            assumptions=["變量間具有線性關係", "殘差呈常態分布", "無多重共線性", "殘差具有同質性"],
            output_metrics=["迴歸係數", "R平方值", "調整後R平方", "F統計量", "VIF值"],
            visualization=["偏迴歸圖", "殘差圖", "係數圖"],
            example={
                "X": [[1, 2], [2, 3], [3, 4]],
                "y": [4, 5, 6]
            }),
        StatisticalMethodInfo(
            method_id="logistic_regression",
            name="邏輯迴歸分析",
            description="預測二元分類結果的機率",
            use_cases=["風險預測", "分類問題", "因素影響力分析"],
            required_data={
                "X": "自變量數值型數據矩陣",
                "y": "二元分類結果（0/1）"
            },
            assumptions=["因變量為二元", "觀察值獨立", "無多重共線性", "樣本量充足"],
            output_metrics=["勝算比", "ROC曲線", "AUC值", "分類準確率", "敏感度和特異度"],
            visualization=["ROC曲線", "混淆矩陣", "係數森林圖"],
            example={
                "X": [[1, 2], [2, 3], [3, 4]],
                "y": [0, 1, 1]
            }),
        StatisticalMethodInfo(method_id="factor_analysis",
                              name="因子分析",
                              description="識別潛在的構念或因子結構",
                              use_cases=["問卷構念效度分析", "維度縮減", "潛在結構探索"],
                              required_data={
                                  "X": "多個變量的數值型數據矩陣",
                                  "feature_names": "變量名稱列表"
                              },
                              assumptions=["變量間具有相關性", "樣本量充足", "數據近似常態分布"],
                              output_metrics=["因子負荷量", "共同性", "特徵值", "解釋變異量"],
                              visualization=["碎石圖", "因子負荷圖", "因子得分散點圖"],
                              example={
                                  "X": [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                                  "feature_names": ["變量1", "變量2", "變量3"]
                              }),
        StatisticalMethodInfo(
            method_id="anova",
            name="變異數分析",
            description="比較三個或以上組別的平均數差異",
            use_cases=["多組處理效果比較", "不同水平間的差異分析", "實驗設計分析"],
            required_data={
                "groups": "多組數值型數據",
                "group_names": "組別名稱列表"
            },
            assumptions=["數據為數值型", "組內數據常態分布", "組間變異數相等", "觀察值獨立"],
            output_metrics=["F 統計量", "p 值", "效果量（Eta-squared）", "組間比較結果"],
            visualization=["箱型圖", "小提琴圖", "均值誤差圖"],
            example={
                "groups": [[1.2, 2.3, 3.1], [1.8, 2.5, 3.0], [2.1, 2.8, 3.3]],
                "group_names": ["組別1", "組別2", "組別3"]
            }),
        StatisticalMethodInfo(method_id="chi_square",
                              name="卡方檢定",
                              description="分析類別變數之間的關聯性或適合度",
                              use_cases=["獨立性檢定", "適合度檢定", "列聯表分析"],
                              required_data={
                                  "observed": "觀察值頻次",
                                  "expected": "期望值頻次"
                              },
                              assumptions=["數據為類別型", "期望頻次足夠大", "觀察值獨立"],
                              output_metrics=["卡方統計量", "p 值", "自由度", "效果量"],
                              visualization=["長條圖", "馬賽克圖", "貢獻值圖"],
                              example={
                                  "observed": [10, 20, 30, 40],
                                  "expected": [15, 25, 35, 25]
                              }),
        StatisticalMethodInfo(
            method_id="survival",
            name="存活分析",
            description="分析時間到事件發生的數據",
            use_cases=["生存時間分析", "失效時間分析", "復發時間分析"],
            required_data={
                "durations": "時間數據",
                "events": "事件指標（0/1）",
                "groups": "分組變數",
                "group_names": "組別名稱"
            },
            assumptions=["觀察值獨立", "審查機制獨立", "風險比例假設"],
            output_metrics=["中位存活時間", "存活率", "風險比", "log-rank 檢定"],
            visualization=["存活曲線", "累積風險圖", "風險比圖"],
            example={
                "durations": [10.2, 15.3, 20.1, 25.4, 30.2],
                "events": [1, 1, 0, 1, 0],
                "groups": [1, 1, 2, 2, 2],
                "group_names": ["治療組", "對照組"]
            }),
        StatisticalMethodInfo(method_id="hypothesis",
                              name="假設檢定",
                              description="檢驗關於總體參數的統計假設",
                              use_cases=["單一樣本檢定", "參數檢定", "假設驗證"],
                              required_data={
                                  "data": "樣本數據",
                                  "hypothesis_value": "虛無假設值"
                              },
                              assumptions=["數據為數值型", "樣本具有代表性", "抽樣獨立"],
                              output_metrics=["檢定統計量", "p 值", "置信區間", "效果量"],
                              visualization=["直方圖", "Q-Q圖", "檢定力曲線"],
                              example={
                                  "data": [1.2, 2.3, 3.1, 2.8, 3.5],
                                  "hypothesis_value": 2.5
                              }),
        StatisticalMethodInfo(method_id="pca",
                              name="主成分分析",
                              description="降維和特徵提取的統計方法",
                              use_cases=["維度縮減", "特徵提取", "數據壓縮"],
                              required_data={
                                  "X": "多變量數據矩陣",
                                  "feature_names": "特徵名稱列表",
                                  "n_components": "要保留的主成分數量"
                              },
                              assumptions=["變量為數值型", "變量間存在相關性", "線性關係假設"],
                              output_metrics=["特徵值", "解釋變異比例", "載荷矩陣", "得分"],
                              visualization=["碎石圖", "雙標圖", "累積變異圖"],
                              example={
                                  "X": [[1.2, 2.3, 3.1], [2.1, 3.2, 2.9],
                                        [1.8, 2.5, 3.0]],
                                  "feature_names": ["特徵1", "特徵2", "特徵3"],
                                  "n_components":
                                  2
                              })
    ]
