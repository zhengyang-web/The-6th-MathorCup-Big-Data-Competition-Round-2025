import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟数据
np.random.seed(42)
n_samples = 2000

data = {
    '保价金额': np.round(np.random.exponential(scale=1000, size=n_samples)),
    '运输距离(km)': np.random.randint(50, 3000, size=n_samples),
    '商品类型': np.random.choice(['电子产品', '易碎品', '服装', '食品', '家具'], size=n_samples, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
    '包装质量': np.random.choice(['普通', '加强', '专业'], size=n_samples, p=[0.6, 0.3, 0.1]),
    '运输方式': np.random.choice(['陆运', '空运', '海运'], size=n_samples, p=[0.7, 0.2, 0.1]),
    '季节': np.random.choice(['春', '夏', '秋', '冬'], size=n_samples),
    '历史理赔次数': np.random.poisson(0.5, size=n_samples),
    '寄件人类型': np.random.choice(['个人', '企业'], size=n_samples, p=[0.65, 0.35]),
    '目的地风险等级': np.random.choice(['低', '中', '高'], size=n_samples, p=[0.7, 0.25, 0.05]),
    '运输时长(天)': np.round(np.random.normal(3, 1, size=n_samples)).clip(1, 10)
}

# 模拟理赔金额（非线性关系）
df = pd.DataFrame(data)
df['理赔金额'] = (
    0.3 * np.log(df['保价金额'] + 1) * df['运输距离(km)'] * 
    np.where(df['商品类型'] == '易碎品', 1.5, 1.0) *
    np.where(df['包装质量'] == '普通', 1.2, 1.0) *
    np.where(df['目的地风险等级'] == '高', 1.8, 1.0) +
    np.random.normal(0, 100, size=n_samples)
).clip(0, 10000)

# 2. 数据预处理
df = pd.get_dummies(df, columns=['商品类型', '包装质量', '运输方式', '季节', '寄件人类型', '目的地风险等级'])
X = df.drop('理赔金额', axis=1)
y = df['理赔金额']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练XGBoost模型
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    reg_lambda=1,
    gamma=0,
    random_state=42
)
model.fit(X_train, y_train)

# 4. 模型评估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

# 5. 可视化分析
# 5.1 特征重要性柱状图
plt.figure(figsize=(12, 8))
plot_importance(model, height=0.8)
plt.title('XGBoost特征重要性排名')
plt.tight_layout()
plt.savefig('q1_特征重要性排名.png')
plt.close()

# 5.2 实际值 vs 预测值散点图
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际理赔金额')
plt.ylabel('预测理赔金额')
plt.title('实际值与预测值对比')
plt.grid(True)
plt.savefig('q1_实际值与预测值对比.png')
plt.close()

# 5.3 残差分布图
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('预测残差分布')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig('q1_预测残差分布.png')
plt.close()

# 5.4 SHAP特征贡献力热力图（添加异常处理）
try:
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10)
    plt.title('SHAP特征贡献力')
    plt.tight_layout()
    plt.savefig('q1_SHAP特征贡献力.png')
    plt.close()
except Exception as e:
    print(f"SHAP分析失败: {e}")
    # 如果SHAP失败，使用内置的特征重要性
    plt.figure(figsize=(12, 8))
    plot_importance(model, height=0.8)
    plt.title('特征重要性排名(替代SHAP)')
    plt.tight_layout()
    plt.savefig('q1_特征重要性排名_替代.png')
    plt.close()

# 5.5 保价金额与理赔金额关系图
plt.figure(figsize=(12, 8))
sns.scatterplot(x='保价金额', y='理赔金额', hue='商品类型_易碎品', 
                size='运输距离(km)', sizes=(20, 200), 
                alpha=0.7, data=df)
plt.title('保价金额与理赔金额关系（按商品类型和运输距离）')
plt.grid(True)
plt.savefig('q1_保价金额与理赔金额关系.png')
plt.close()

# 5.6 运输距离与理赔金额箱线图
df['运输距离分组'] = pd.cut(df['运输距离(km)'], bins=[0, 500, 1000, 1500, 2000, 2500, 3000])
plt.figure(figsize=(12, 8))
sns.boxplot(x='运输距离分组', y='理赔金额', hue='包装质量_普通', data=df)
plt.title('不同运输距离下的理赔金额分布（按包装质量）')
plt.xticks(rotation=45)
plt.savefig('q1_运输距离与理赔金额分布.png')
plt.close()

# 5.7 模型学习曲线
# 注意：需要先设置eval_set才能获取学习曲线
model_with_eval = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    reg_lambda=1,
    gamma=0,
    random_state=42
)
model_with_eval.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
results = model_with_eval.evals_result()

plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['rmse'], label='测试集RMSE')
plt.xlabel('迭代次数')
plt.ylabel('RMSE值')
plt.title('模型学习曲线')
plt.legend()
plt.grid(True)
plt.savefig('q1_模型学习曲线.png')
plt.close()

# 5.8 理赔金额分布雷达图
categories = ['电子产品', '易碎品', '服装', '食品', '家具']
values = [df[df['商品类型_'+c] == 1]['理赔金额'].mean() for c in categories]
values += [values[0]]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += [angles[0]]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_title('不同商品类型的平均理赔金额')
ax.grid(True)
plt.savefig('q1_不同商品类型理赔金额雷达图.png')
plt.close()

# 5.9 交互效应3D图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

xs = df['保价金额']
ys = df['运输距离(km)']
zs = df['理赔金额']

scatter = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', alpha=0.6)
ax.set_xlabel('保价金额')
ax.set_ylabel('运输距离(km)')
ax.set_zlabel('理赔金额')
plt.title('保价金额、运输距离与理赔金额的3D关系')
plt.colorbar(scatter)
plt.savefig('q1_三维交互效应图.png')
plt.close()

# 5.10 时间序列分析（模拟）
df['日期'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
monthly_avg = df.resample('M', on='日期')['理赔金额'].mean()

plt.figure(figsize=(12, 6))
monthly_avg.plot()
plt.xlabel('日期')
plt.ylabel('平均理赔金额')
plt.title('月度平均理赔金额趋势')
plt.grid(True)
plt.savefig('q1_月度理赔金额趋势.png')
plt.close()