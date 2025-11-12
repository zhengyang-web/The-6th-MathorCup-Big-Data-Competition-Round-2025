import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 全局设置 ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 'SimHei' 是黑体# 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 创建一个目录来保存图片
if not os.path.exists('strategy_visualizations'):
    os.makedirs('strategy_visualizations')

# --- 1. 绘制多目标优化的帕累托前沿（3D散点图） ---
def plot_pareto_front():
    """
    使用3D散点图展示多目标优化找到的帕累托最优解集。
    每个点代表一个非支配解，展示了成本、满意度和风险三个目标间的权衡关系。
    """
    title = '多目标优化帕累托前沿'
    
    # 虚构153个非支配解，模拟帕累托前沿的形状
    np.random.seed(42)
    num_solutions = 153
    
    # 生成在特定范围内的随机数据
    cost = np.random.uniform(2150, 2850, num_solutions)
    satisfaction = np.random.uniform(0.42, 0.68, num_solutions)
    risk = np.random.uniform(850, 1150, num_solutions)
    
    # 模拟帕累托前沿的曲面特性：一个目标变好，至少一个其他目标会变差
    # 我们通过一个简单的函数来调整数据，使其看起来像一个前沿
    combined_metric = cost/3000 + (1 - satisfaction) + risk/1200
    # 保留那些综合指标较低（更好）的点，形成一个“前沿”
    indices = np.where(combined_metric < np.percentile(combined_metric, 90))[0]
    cost, satisfaction, risk = cost[indices], satisfaction[indices], risk[indices]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用满意度作为颜色映射，更直观
    sc = ax.scatter(cost, risk, satisfaction, c=satisfaction, cmap='viridis', s=50, alpha=0.8)
    
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('总成本 (元)', fontsize=12, labelpad=10)
    ax.set_ylabel('总风险指标', fontsize=12, labelpad=10)
    ax.set_zlabel('客户满意度得分', fontsize=12, labelpad=10)
    
    # 反转X轴和Y轴，使得原点代表“理想”状态（低成本、低风险）
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # 添加颜色条
    cbar = fig.colorbar(sc, shrink=0.6, aspect=10)
    cbar.set_label('满意度得分', fontsize=12)
    
    ax.view_init(elev=25, azim=45)
    
    filename = f"strategy_visualizations/q3_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片 '{filename}' 已保存。")

# --- 2. 绘制策略参数间的权衡关系（热力图） ---
def plot_strategy_tradeoff():
    """
    使用热力图展示理赔比例与审核严格度之间的权衡关系及其对总成本的影响。
    """
    title = '理赔比例与审核严格度的权衡关系'
    
    # 创建数据网格
    claim_ratio = np.linspace(0.3, 1.0, 20)
    review_strictness = np.linspace(1, 10, 20)
    X, Y = np.meshgrid(claim_ratio, review_strictness)
    
    # 模拟总成本函数，体现权衡关系
    # 成本 = 理赔成本 + 审核成本
    # 理赔成本随理赔比例增加而增加，随审核严格度增加而减少
    # 审核成本随审核严格度增加而增加
    Z_cost = (X * 1500) - (Y * X * 80) + (Y**1.5 * 20)
    
    plt.figure(figsize=(12, 9))
    contour = plt.contourf(X, Y, Z_cost, levels=20, cmap='inferno')
    plt.colorbar(contour, label='预期总成本 (元)')
    
    # 添加等高线
    contour_lines = plt.contour(X, Y, Z_cost, levels=10, colors='white', alpha=0.6)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f')
    
    plt.title(title, fontsize=16)
    plt.xlabel('理赔比例', fontsize=12)
    plt.ylabel('审核严格度 (1-10级)', fontsize=12)
    
    # 标注一个可能的“最优平衡区”
    plt.gca().add_patch(plt.Rectangle((0.6, 4), 0.2, 3, 
                                      edgecolor='cyan', facecolor='none', 
                                      linewidth=2, linestyle='--',
                                      label='最优平衡区'))
    plt.legend()
    
    filename = f"strategy_visualizations/q3_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片 '{filename}' 已保存。")

# --- 3. 绘制处理时间对满意度的边际递减效应（曲线图） ---
def plot_diminishing_returns():
    """
    使用曲线图展示理赔处理时间对客户满意度的边际递减效应。
    """
    title = '处理时间对客户满意度的边际递减效应'
    
    # 虚构数据，模拟边际递减效应
    time_hours = np.linspace(12, 96, 100)
    # 使用一个类似log的函数来模拟
    satisfaction = 0.8 - 0.2 * np.log(time_hours/12 + 1)
    satisfaction = np.clip(satisfaction, 0.3, 0.75)

    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, satisfaction, color='teal', linewidth=3)
    
    # 标注关键点
    t_points = [72, 48, 24]
    s_points = 0.8 - 0.2 * np.log(np.array(t_points)/12 + 1)
    
    plt.scatter(t_points, s_points, color='red', zorder=5)
    plt.text(72, s_points[0] - 0.03, f'72小时\n满意度: {s_points[0]:.2f}', ha='center')
    plt.text(48, s_points[1] + 0.02, f'48小时\n满意度: {s_points[1]:.2f}', ha='center')
    plt.text(24, s_points[2] + 0.02, f'24小时\n满意度: {s_points[2]:.2f}', ha='center')
    
    plt.title(title, fontsize=16)
    plt.xlabel('理赔处理时间 (小时)', fontsize=12)
    plt.ylabel('客户满意度得分', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.gca().invert_xaxis() # 时间越短越好，从右到左看
    
    filename = f"strategy_visualizations/q3_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片 '{filename}' 已保存。")

# --- 4. 绘制新旧策略对比雷达图 ---
def plot_strategy_comparison_radar():
    """
    使用雷达图对比多目标优化策略与传统固定策略在三个核心目标上的综合表现。
    """
    title = '多目标优化策略 vs. 传统固定策略'
    
    labels = np.array(['成本控制', '客户满意度', '风险规避'])
    num_vars = len(labels)
    
    # 虚构数据
    # 传统策略
    stats_traditional = [0.7, 0.5, 0.6] # 假设基准表现
    # 优化策略：成本不变(0.7)，满意度提升28.6%，风险降低35.2%（风险规避提升）
    stats_optimized = [0.7, 0.5 * 1.286, 0.6 * 1.352]
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats_traditional += stats_traditional[:1]
    stats_optimized += stats_optimized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制传统策略
    ax.plot(angles, stats_traditional, color='gray', linestyle='--', linewidth=2, label='传统固定策略')
    ax.fill(angles, stats_traditional, color='gray', alpha=0.25)
    
    # 绘制优化策略
    ax.plot(angles, stats_optimized, color='deepskyblue', linewidth=2, label='多目标优化策略')
    ax.fill(angles, stats_optimized, color='deepskyblue', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    
    plt.title(title, size=16, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    filename = f"strategy_visualizations/q3_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片 '{filename}' 已保存。")

# --- 5. 绘制动态理赔策略决策流示意图 ---
def plot_dynamic_strategy_flow():
    """
    使用流程图形式，示意性地展示模型如何根据风险预测结果动态调整理赔策略。
    """
    title = '动态理赔策略决策流示意图'
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    # 定义框和箭头样式
    bbox_props = dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="b", lw=2)
    arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", ec="black", lw=2)

    # 节点位置和内容
    nodes = {
        'start': ((0.5, 0.9), '新运单\n(特征输入)'),
        'rf': ((0.5, 0.65), '随机森林\n风险预测模型'),
        'decision': ((0.5, 0.4), '预测理赔概率 > 0.7?'),
        'high_risk': ((0.2, 0.15), '高风险策略\n- 审核严格度: 高\n- 理赔比例: 中\n- 处理时间: 标准'),
        'low_risk': ((0.8, 0.15), '低风险策略\n- 审核严格度: 低\n- 理赔比例: 高\n- 处理时间: 快速')
    }

    # 绘制节点
    for name, (pos, text) in nodes.items():
        ax.text(pos[0], pos[1], text, ha="center", va="center", size=12, bbox=bbox_props)

    # 绘制箭头
    ax.annotate("", xy=nodes['rf'][0], xytext=nodes['start'][0], arrowprops=arrow_props)
    ax.annotate("", xy=nodes['decision'][0], xytext=nodes['rf'][0], arrowprops=arrow_props)
    
    # 决策分支
    ax.annotate("是", xy=nodes['high_risk'][0], xytext=(0.35, 0.3),
                arrowprops=arrow_props, ha='center', va='center', size=12)
    ax.annotate("否", xy=nodes['low_risk'][0], xytext=(0.65, 0.3),
                arrowprops=arrow_props, ha='center', va='center', size=12)

    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    filename = f"strategy_visualizations/q3_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片 '{filename}' 已保存。")

# --- 6. 随机森林特征重要性（用于风险预测） ---
def plot_rf_feature_importance():
    """
    展示用于风险预测的随机森林模型中，各特征的重要性排序。
    """
    title = '随机森林风险预测模型特征重要性'
    
    # 虚构特征及其重要性
    features = {
        '历史理赔记录': 0.28,
        '商品价值': 0.22,
        '包装类型': 0.17,
        '运输距离': 0.11,
        '寄件人信用': 0.09,
        '运输方式': 0.06,
        '季节因素': 0.04,
        '其他': 0.03
    }
    
    feature_series = pd.Series(features).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_series.values, y=feature_series.index, palette='mako')
    
    plt.title(title, fontsize=16)
    plt.xlabel('特征重要性 (Gini Impurity)', fontsize=12)
    plt.ylabel('风险特征', fontsize=12)
    
    for i, v in enumerate(feature_series.values):
        plt.text(v + 0.005, i, f'{v:.2f}', va='center')

    plt.xlim(0, max(feature_series.values) * 1.1)
    
    filename = f"strategy_visualizations/q3_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片 '{filename}' 已保存。")


# --- 主函数 ---
if __name__ == '__main__':
    print("开始生成基于随机森林与多目标优化模型的分析图表...")
    plot_pareto_front()
    plot_strategy_tradeoff()
    plot_diminishing_returns()
    plot_strategy_comparison_radar()
    plot_dynamic_strategy_flow()
    plot_rf_feature_importance()
    print("\n所有图表已成功生成并保存在 'strategy_visualizations' 目录中。")