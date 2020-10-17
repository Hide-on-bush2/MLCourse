import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl


def read_data(filename):
    df = pd.read_csv(filename)
    print(df)
    return df

def draw_hist(data):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(data["单价"], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Interval")
    plt.ylabel("Frequency/frequency")
    plt.title("Frequency/frequency distribution histogram")
    plt.show()

def draw_box(data):
    # dic = {}
    # for item in data["区"]:
    #     if item in dic.keys():
    #         continue
    #     else:
    #         dic[item] = 1
    #         print(item)
    yantian = data.loc[data.loc[:,"区"]=="盐田", "总价"]
    nanshan = data.loc[data.loc[:,"区"]=="南山", "总价"]
    baoan = data.loc[data.loc[:,"区"]=="宝安", "总价"]
    longgang = data.loc[data.loc[:,"区"]=="龙岗", "总价"]
    longhua = data.loc[data.loc[:,"区"]=="龙华", "总价"]
    futian = data.loc[data.loc[:,"区"]=="福田", "总价"]
    luohu = data.loc[data.loc[:,"区"]=="罗湖", "总价"]
    # print(yantian.describe())
    labels = ["yantain", "nanshan", "baoan", "longgang", "longhua", "futian", "luohu"]
    plt.boxplot([yantian, nanshan, baoan, longgang, longhua, futian, luohu], labels=labels)
    plt.show()

def draw_heatmap(data, index, columns, values):
    pt = data.pivot_table(index=index, columns=columns, values=values, aggfunc=np.sum)
    print(pt)
    f, (ax) = plt.subplots(figsize=(6,6),nrows=1)
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    sns.heatmap(pt, linewidths = 0.05, ax=ax, vmax=900, vmin=0, cmap=cmap)
    plt.show()


def draw_hist2(data):
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
    x = data["总价"]
    ax0.hist(x,100,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75) 
    ax1.hist(x,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
    plt.show()

if __name__ == "__main__":
    data = read_data("./data/shenzhenhouse.csv")
    # draw_hist(data)
    # draw_box(data)
    # draw_heatmap(data, "区", "朝向", "总价")
    draw_hist2(data)
    

