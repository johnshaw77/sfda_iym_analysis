import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from typing import List, Tuple, Optional
from matplotlib.font_manager import FontProperties


def get_chinese_font():
    """獲取中文字體"""
    chinese_fonts = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/Library/Fonts/Microsoft/PMingLiU.ttf',
        '/Library/Fonts/Microsoft/SimSun.ttf',
        '/Library/Fonts/Microsoft/Microsoft JhengHei.ttf'
    ]

    for font_path in chinese_fonts:
        try:
            font = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = [font.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            return font
        except:
            continue

    try:
        plt.rcParams['font.sans-serif'] = [
            'Microsoft JhengHei', 'SimHei', 'Arial Unicode MS'
        ]
        plt.rcParams['axes.unicode_minus'] = False
        return FontProperties(family='Microsoft JhengHei')
    except:
        print("無法載入中文字體，將使用預設字體")
        return None


def get_base64_plot():
    """將當前的 matplotlib 圖形轉換為 base64 字符串"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    return graphic.decode('utf-8')


def create_scatter_plot(x: List[float],
                        y: List[float],
                        title: str = "散點圖",
                        xlabel: str = "X",
                        ylabel: str = "Y") -> str:
    """創建散點圖"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return get_base64_plot()


def create_regression_plot(x: List[float],
                           y: List[float],
                           coef: float,
                           intercept: float,
                           title: str = "回歸分析",
                           xlabel: str = "X",
                           ylabel: str = "Y") -> str:
    """創建回歸圖"""
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.5})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return get_base64_plot()


def create_box_plot(data: List[List[float]],
                    labels: List[str],
                    title: str = "箱型圖",
                    xlabel: str = "組別",
                    ylabel: str = "數值") -> str:
    """創建箱型圖"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return get_base64_plot()


def create_histogram(data: List[float],
                     title: str = "直方圖",
                     xlabel: str = "數值",
                     ylabel: str = "頻率") -> str:
    """創建直方圖"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return get_base64_plot()


def create_qq_plot(data: List[float], title: str = "Q-Q 圖") -> str:
    """創建 Q-Q 圖"""
    plt.figure(figsize=(10, 6))
    from scipy import stats
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    return get_base64_plot()


def create_residual_plot(x: List[float],
                         residuals: List[float],
                         title: str = "殘差圖",
                         xlabel: str = "預測值",
                         ylabel: str = "殘差") -> str:
    """創建殘差圖"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return get_base64_plot()
