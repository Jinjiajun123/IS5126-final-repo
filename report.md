# 社交电商购买预测与卖家优化系统技术报告

## 1. 项目概述

### 1.1 项目背景

在社交电商场景中，卖家最关心的问题并不是“用户最终买没买”，而是：

- 商品刚上架时，这个 listing 是否具备较高转化潜力？
- 当用户已经进入商品浏览会话后，当前这次会话是否大概率会转化？
- 卖家可以优先优化哪些可控因素，例如标题、图片数、价格、折扣、视频等？

因此，本项目不再把“购买预测”看作单一任务，而是将其拆分为两个具有不同业务时点的预测任务，并配套聚类分层与可解释建议模块。

### 1.2 项目目标

本项目的目标包括：

1. 构建两类购买预测模型：
   - `listing_time`：商品上架时预测
   - `session_time`：会话过程中预测
2. 明确区分真实可部署特征、会话行为特征和禁止使用的泄漏特征。
3. 使用统一的训练管线保证训练、评估和推理一致性。
4. 通过用户聚类构建商家可理解的人群画像与运营策略建议。
5. 为前端诊断页面提供可落地的概率输出、混合评分和优化建议。

### 1.3 方法总览

当前系统由以下几个部分组成：

- 双任务监督学习：
  - `listing_time`
  - `session_time`
- 特征分层与 leakage 控制
- `Pipeline + ColumnTransformer` 统一预处理
- 三种评估切分：
  - 分层随机切分
  - 时间代理切分
  - 用户分组切分
- 概率校准
- 阈值调优
- 用户聚类对比实验：
  - `KMeans`
  - `HDBSCAN`
- 规则评分与混合评分：
  - `domain score`
  - `hybrid score`

---

## 2. 数据集说明

### 2.1 数据概况

原始数据文件为 [social_ecommerce_data.csv](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/social_ecommerce_data.csv)。

数据概况如下：

| 指标 | 数值 |
|---|---|
| 样本量 | 100,000 |
| 原始特征数 | 32 |
| 目标变量 | `label` |
| 正类含义 | `1 = 购买` |
| 购买率 | 44.98% |

### 2.2 特征分组

为了适配双任务建模，本项目将特征分为四类：

#### 2.2.1 卖家可控特征

- `title_length`
- `title_emo_score`
- `img_count`
- `has_video`
- `price`
- `discount_rate`

这些特征可直接被卖家优化，是 `listing_time` 模型的核心。

#### 2.2.2 静态用户上下文特征

- `age`
- `gender`
- `user_level`
- `purchase_freq`
- `total_spend`
- `register_days`
- `follow_num`
- `fans_num`

这些特征相对稳定，适合作为用户画像背景信息。

#### 2.2.3 会话行为特征

- `is_follow_author`
- `add2cart`
- `coupon_received`
- `coupon_used`
- `pv_count`
- `last_click_gap`
- `purchase_intent`
- `freshness_score`

这些特征仅适用于 `session_time` 任务，因为它们通常需要用户已经进入商品浏览或交互阶段后才可观察到。

#### 2.2.4 禁止使用的泄漏特征

- `like_num`
- `comment_num`
- `share_num`
- `collect_num`
- `interaction_rate`
- `social_influence`

这些特征属于典型的 post-event / post-exposure 信号，若直接用于训练，会高估模型效果，破坏部署可行性。

---

## 3. 任务重定义与数据泄漏控制

### 3.1 为什么需要重定义任务

项目早期最大的问题是：将“商品刚上架时的优化预测”和“用户浏览过程中是否转化的预测”混在一起建模。

这种做法会导致：

- 训练时使用了上线时实际拿不到的行为信号
- 模型离线指标偏高，但部署时无法复现
- 系统无法明确说明“到底在什么时点进行预测”

因此，本项目将预测任务拆分为：

#### `listing_time`

定义：在商品刚创建或刚发布时，对其转化潜力进行评分。

允许使用：

- 卖家可控特征
- 静态用户画像
- 类目特征

不允许使用：

- 会话行为特征
- 所有泄漏特征

#### `session_time`

定义：在用户已经进入商品浏览会话之后，预测这次会话是否会转化。

允许使用：

- `listing_time` 的全部特征
- 会话行为特征

不允许使用：

- 泄漏特征

### 3.2 数据泄漏控制策略

本项目主要防范三类 leakage：

1. **Target Leakage**
   - 例如互动量、收藏、分享等结果后形成的特征
2. **Train-Test Contamination**
   - 例如在切分前先做全量标准化
3. **Temporal Leakage**
   - 例如用未来样本的信息帮助预测过去

本项目通过以下方式控制 leakage：

- 明确删除禁止特征
- 使用 `Pipeline` 让预处理只在训练集上拟合
- 引入时间代理切分与用户分组切分
- 将任务定义与特征边界写入 artifacts，供训练和推理共同使用

---

## 4. 建模流程

### 4.1 统一训练管线

训练逻辑现已统一收敛到 [training_pipeline.py](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/training_pipeline.py)。

整体流程如下：

1. 读取原始数据
2. 构造 `event_index` 作为时间代理顺序
3. 按任务选择特征集
4. 用 `ColumnTransformer` 完成预处理
5. 用 `RandomForestClassifier` 训练分类模型
6. 用 `CalibratedClassifierCV` 进行概率校准
7. 输出 split 评估结果
8. 训练全量最终模型
9. 训练并保存聚类模块
10. 生成 `model_artifacts.pkl`、`processed_data.csv`、`model_metrics.json`

### 4.2 预处理策略

当前统一预处理采用：

- 数值列：
  - `SimpleImputer(strategy="median")`
- 类别列：
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

这样可以保证：

- 缺失值处理稳定
- 类目特征不引入伪序关系
- 训练与推理特征处理保持一致

### 4.3 主模型

当前主模型使用：

- `RandomForestClassifier`

主要配置包括：

- `n_estimators=300`
- `max_depth=10`
- `min_samples_leaf=5`
- `class_weight="balanced_subsample"`
- `random_state=42`

选择随机森林的原因：

- 能处理非线性关系
- 对特征缩放不敏感
- 与当前表格型数据适配度高
- 易于作为稳定基线模型部署

### 4.4 概率校准

由于本项目不仅输出类别，还需要输出：

- 购买概率
- 加权 cluster 概率
- hybrid score
- counterfactual 概率差值

因此单纯使用原始 `predict_proba` 不足够。  
本项目使用：

- `CalibratedClassifierCV(method="sigmoid")`

其目标不是提升 AUC，而是提升概率质量，主要通过 `Brier score` 观察。

### 4.5 阈值调优

在校准概率基础上，本项目进一步加入 threshold tuning。

做法是：

1. 在训练集内部再切出一个验证子集
2. 在验证子集上搜索阈值
3. 以 `F1` 为目标寻找最优 threshold
4. 将该 threshold 用于测试集评估

需要注意的是：

- 这种方式对 `session_time` 是有效的
- 对 `listing_time` 则会出现极低阈值、近乎全判正类的问题

因此当前 `listing_time` 的 tuned threshold 结果仅作为实验结果，不建议直接用于线上分类决策。

---

## 5. 评估方案

### 5.1 三种切分方式

为了避免只依赖单一切分导致的乐观估计，本项目对每个任务都进行三种评估：

#### 5.1.1 分层随机切分

- 方法：`StratifiedShuffleSplit`
- 作用：作为标准机器学习基线评估

#### 5.1.2 时间代理切分

- 按 `event_index` 前 80% / 后 20% 切分
- 由于当前数据集中没有显式时间戳，因此将原始样本顺序作为时间代理
- 作用：模拟未来样本上的泛化能力

#### 5.1.3 用户分组切分

- 方法：`GroupShuffleSplit`
- 分组键：`user_id`
- 作用：避免同一用户同时出现在训练集和测试集中，评估跨用户泛化能力

### 5.2 评估指标

本项目使用以下指标：

- `ROC-AUC`
- `PR-AUC`
- `F1`
- `Precision`
- `Recall`
- `Brier Score`
- `Positive Rate`

其中：

- `ROC-AUC / PR-AUC` 用于评估区分能力
- `Brier Score` 用于评估概率质量
- `F1 / Precision / Recall` 用于评估分类效果

---

## 6. 模型结果分析

结果基于 [model_metrics.json](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/model_metrics.json)。

### 6.1 `listing_time` 模型结果

#### 分层随机切分

- Raw ROC-AUC: `0.5742`
- Calibrated ROC-AUC: `0.5742`
- Calibrated F1: `0.3612`
- Tuned F1: `0.6199`
- Tuned Threshold: `0.3250`

#### 时间代理切分

- Raw ROC-AUC: `0.5753`
- Calibrated ROC-AUC: `0.5774`
- Calibrated F1: `0.3513`
- Tuned F1: `0.6181`
- Tuned Threshold: `0.2750`

#### 用户分组切分

- Raw ROC-AUC: `0.5749`
- Calibrated ROC-AUC: `0.5741`
- Calibrated F1: `0.3574`
- Tuned F1: `0.6220`
- Tuned Threshold: `0.1000`

#### 结果解读

`listing_time` 模型表现整体较弱，但这是符合任务定义的。

原因在于该模型严格限制在“上架时可获得”的特征范围内，仅依赖：

- 卖家可控属性
- 静态用户背景
- 类目信息

在这种设定下，ROC-AUC 约 `0.57` 说明模型具备一定排序能力，但尚不足以作为高精度购买分类器。

更重要的是，当前 tuned threshold 虽然大幅提高了 F1，但这是通过极低阈值实现的，带来了：

- 非常高的 `positive_rate`
- 接近 `1.0` 的 `recall`
- 较低的 `precision`

因此：

- `listing_time` 更适合做排序、打分、商品优化诊断
- 不适合直接作为硬分类器使用
- 当前 tuned threshold 不建议直接用于业务上线

### 6.2 `session_time` 模型结果

#### 分层随机切分

- Raw ROC-AUC: `0.7711`
- Calibrated ROC-AUC: `0.7711`
- Calibrated F1: `0.6560`
- Tuned F1: `0.6854`
- Tuned Threshold: `0.3116`

#### 时间代理切分

- Raw ROC-AUC: `0.7676`
- Calibrated ROC-AUC: `0.7672`
- Calibrated F1: `0.6481`
- Tuned F1: `0.6810`
- Tuned Threshold: `0.3250`

#### 用户分组切分

- Raw ROC-AUC: `0.7743`
- Calibrated ROC-AUC: `0.7743`
- Calibrated F1: `0.6525`
- Tuned F1: `0.6897`
- Tuned Threshold: `0.3368`

#### 结果解读

`session_time` 模型显著强于 `listing_time`。

这说明在会话阶段，以下行为特征具有显著增益：

- `add2cart`
- `coupon_received`
- `coupon_used`
- `pv_count`
- `purchase_intent`
- `last_click_gap`

同时，三种切分下结果稳定，说明该模型：

- 没有明显依赖特定 split 才表现好
- 对不同评估方式具有较好的鲁棒性

Threshold tuning 在 `session_time` 上是有效的：

- tuned threshold 大约稳定在 `0.31 - 0.34`
- tuned F1 相比 calibrated F1 有稳定提升

因此，`session_time` 可视为当前项目中真正成熟、可作为分类器使用的预测模型。

### 6.3 概率校准效果

校准的主要收益体现在 `Brier score` 上。

例如：

- `listing_time`
  - raw brier 约 `0.2445 - 0.2448`
  - calibrated brier 约 `0.2421 - 0.2424`
- `session_time`
  - raw brier 约 `0.1921 - 0.1944`
  - calibrated brier 约 `0.1906 - 0.1931`

这说明校准确实改善了概率输出质量。

对于本项目来说，这一点非常重要，因为后续模块依赖概率值进行：

- cluster 加权
- hybrid score
- counterfactual lift 分析

---

## 7. 用户聚类实验

### 7.1 聚类目标

本项目中的聚类不用于替代监督学习，而是用于：

- 构建可解释的人群画像
- 支撑运营策略建议
- 帮助商家理解“哪些类型用户更值得关注”

因此，我们将其定义为 **persona clustering**，而不是 session clustering。

### 7.2 聚类实验设计

本项目比较了以下内容：

#### 7.2.1 特征集

1. `persona_baseline`
   - 基础用户画像特征
2. `persona_value_focus`
   - 更强调消费能力与价值
3. `persona_demographic_plus_value`
   - 兼顾人口属性与价值特征

#### 7.2.2 特征变换

- 原始数值 + 标准化
- `log1p + 标准化`

#### 7.2.3 算法

- `KMeans`
- `HDBSCAN`

其中 `KMeans` 用于生成固定簇数、适合前端展示的人群分层；  
`HDBSCAN` 用于检查是否存在更适合密度建模的用户结构和噪声用户。

### 7.3 聚类结果

本轮实验中，统计上最佳的聚类配置为：

- 算法：`HDBSCAN`
- 特征集：`persona_demographic_plus_value`
- `use_log=False`
- 结果：
  - silhouette: `0.2328`
  - cluster_count: `2`
  - noise_rate: `0.0987`
  - size_balance: `0.5447`

作为对比，较强的 `KMeans` 配置包括：

- `persona_demographic_plus_value + KMeans(k=4)`
  - silhouette: `0.2208`

### 7.4 聚类结果解读

这一结果表明：

1. 新的人群特征设计优于早期版本的简单 baseline。
2. 数据中确实存在一定比例的噪声用户，约为 9% 到 10%。
3. `HDBSCAN` 在统计指标上略优于 `KMeans`。

但从业务角度看，仍然存在权衡：

- `HDBSCAN`
  - 优点：统计指标更高，允许噪声点存在
  - 缺点：最终只有 2 个 cluster，业务粒度较粗
- `KMeans(k=4)`
  - 优点：更适合做多类 persona 卡片和商家运营建议
  - 缺点：统计指标略低

因此，本项目对聚类的结论是：

- 在统计层面，`HDBSCAN` 是当前更优的实验结果
- 在产品化和运营可解释性层面，`KMeans(k=4)` 仍然具有展示与沟通优势

---

## 8. 系统运行逻辑

### 8.1 训练侧

训练由 [training_pipeline.py](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/training_pipeline.py) 统一执行。

训练输出包括：

- [model_artifacts.pkl](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/model_artifacts.pkl)
- [processed_data.csv](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/processed_data.csv)
- [model_metrics.json](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/model_metrics.json)

### 8.2 推理侧

后端位于 [backend/main.py](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/backend/main.py)。

当前主要支持：

- `/api/analyze`
  - 默认走 `listing_time`
- `/api/evaluate_hybrid`
  - 默认走 `session_time`
  - 同时结合 domain score 与外部多模态诊断能力

### 8.3 前端与展示

前端包括：

- FastAPI 后端接口
- Vue 前端诊断页
- Streamlit 原型页

这些页面共享同一份 artifacts，因此不会出现 notebook、训练脚本、后端三套逻辑不一致的问题。

---

## 9. 项目结论

### 9.1 主要结论

本项目的核心结论如下：

1. **购买预测必须区分业务时点。**
   将 `listing_time` 与 `session_time` 拆开后，模型能力与部署可行性得到了更清晰的解释。

2. **`listing_time` 模型是弱信号模型，但具备真实部署意义。**
   它适合做上架前打分、排序与优化建议，不适合直接做强分类决策。

3. **`session_time` 模型是当前最强的预测模型。**
   会话行为特征显著提升了购买预测能力，并且在多种 split 下保持稳定。

4. **概率校准是必要的。**
   即使 AUC 提升有限，`Brier score` 的改善使得模型输出更适合用于概率型业务决策。

5. **threshold tuning 需要区分任务使用。**
   在 `session_time` 上有效，在 `listing_time` 上当前存在过度偏向正类的问题。

6. **聚类更适合作为辅助解释模块，而不是主预测模块。**
   当前最优聚类结构并不非常强，但足以支撑粗粒度 persona 分层与运营建议。

### 9.2 当前最可落地的系统定位

从当前结果来看，最合理的系统定位是：

- `listing_time`：
  商品上架前优化评分与卖家诊断器
- `session_time`：
  会话期购买转化预测器
- 聚类模块：
  用户画像解释层与营销建议支撑模块

---

## 10. 局限性与后续工作

### 10.1 当前局限

1. **原始数据中缺少显式时间戳字段**
   当前时间切分采用 `event_index` 代理，仍弱于真正的时序评估。

2. **`listing_time` 的信号强度有限**
   仅使用上架时特征时，预测能力天然较弱。

3. **threshold tuning 的目标函数仍较单一**
   当前以 F1 为目标，尚未加入 precision 约束或业务收益约束。

4. **聚类统计结构仍不够强**
   尽管新方案有所提升，但 silhouette 整体仍处于中低水平。

5. **标题语义信息不足**
   当前仅使用 `title_length` 和 `title_emo_score`，没有加入真正的标题 embedding。

### 10.2 后续优化方向

1. 引入真实时间字段，做严格的 time-based split 与 drift analysis。
2. 对 `listing_time` 引入标题 embedding、类目文本信息或图片质量特征。
3. 将 threshold tuning 从单纯 F1 优化改为：
   - 加 precision 约束
   - 加 positive rate 约束
   - 或直接基于业务收益优化
4. 对 `session_time` 增加 SHAP 或特征重要性解释。
5. 将聚类结果进一步转化为更可执行的商家策略模板。

---

## 11. 复现实验方法

在项目根目录执行：

```bash
pip install -r requirements.txt
python training_pipeline.py
```

随后可打开：

- [analysis.ipynb](/Users/macbookpro/Desktop/5126final/IS5126-final-repo/analysis.ipynb)

查看：

- 双任务评估结果
- threshold tuning 对比
- 聚类实验表
- 最终 cluster profiles 与策略建议

---

## 12. 附录：当前核心结果摘要

### `listing_time`

| Split | ROC-AUC | Calibrated F1 | Tuned F1 | Tuned Threshold |
|---|---:|---:|---:|---:|
| Stratified Random | 0.5742 | 0.3612 | 0.6199 | 0.3250 |
| Time-Based Proxy | 0.5774 | 0.3513 | 0.6181 | 0.2750 |
| Group-Based User | 0.5741 | 0.3574 | 0.6220 | 0.1000 |

说明：Tuned F1 提升显著，但当前阈值过低，存在近乎全判正类问题，因此不建议直接采用。

### `session_time`

| Split | ROC-AUC | Calibrated F1 | Tuned F1 | Tuned Threshold |
|---|---:|---:|---:|---:|
| Stratified Random | 0.7711 | 0.6560 | 0.6854 | 0.3116 |
| Time-Based Proxy | 0.7672 | 0.6481 | 0.6810 | 0.3250 |
| Group-Based User | 0.7743 | 0.6525 | 0.6897 | 0.3368 |

说明：`session_time` 模型表现稳定，threshold tuning 带来了合理且可解释的提升。
