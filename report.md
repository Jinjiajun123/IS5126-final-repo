# Social E-commerce Purchase Prediction & Seller Recommendation System

## Technical Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Preprocessing & Feature Engineering](#3-data-preprocessing--feature-engineering)
4. [Model Construction & Comparison](#4-model-construction--comparison)
5. [Model Selection & Training Details](#5-model-selection--training-details)
6. [Model Interpretability](#6-model-interpretability)
7. [Sentiment Analysis Pipeline](#7-sentiment-analysis-pipeline)
8. [User Clustering](#8-user-clustering)
9. [Recommendation Engine](#9-recommendation-engine)
10. [System Architecture & Pipeline](#10-system-architecture--pipeline)
11. [Dashboard & Result Presentation](#11-dashboard--result-presentation)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Project Overview

### 1.1 Problem Statement

In social e-commerce platforms, sellers face the challenge of optimizing product listings to maximize purchase conversion rates. The key question is: **given a product's attributes (title, images, price, discount, video, etc.), what actionable changes can a seller make to improve the purchase probability?**

### 1.2 Objectives

1. Build a purchase prediction model that accurately classifies whether a user-product interaction leads to a purchase.
2. Identify the most impactful seller-controllable features through interpretability analysis.
3. Segment users into meaningful clusters for targeted marketing.
4. Provide actionable, data-driven recommendations for sellers through an interactive dashboard.

### 1.3 Approach

The project adopts a full-stack data science pipeline:

- **Data Preprocessing**: Data leakage identification, feature engineering, and stratified train-test split.
- **Multi-Model Comparison**: Logistic Regression, Random Forest, and XGBoost trained and evaluated on the same split.
- **Model Interpretability**: SHAP values, Partial Dependence Plots (PDP), and Counterfactual Analysis.
- **Sentiment Analysis**: SnowNLP-based Chinese text sentiment scoring with Beta quantile calibration.
- **User Clustering**: K-Means segmentation for user persona analysis.
- **Hybrid Scoring**: Combining ML prediction with domain knowledge rules.
- **Interactive Dashboard**: Streamlit-based product analyzer with seller recommendations.

---

## 2. Dataset Description

### 2.1 Overview

| Attribute | Value |
|---|---|
| Source | Social e-commerce platform interaction logs |
| Total Records | 100,000 |
| Total Features | 32 (raw) |
| Target Variable | `label` (1 = purchased, 0 = not purchased) |
| Purchase Rate | 44.98% (44,983 positive / 55,017 negative) |
| Categories | 6 (服饰鞋包, 数码家电, 食品生鲜, 美妆个护, 家居日用, 其他) |

### 2.2 Feature Dictionary

The 32 raw features are organized into the following groups:

**ID Features (2)** - Dropped before modeling:
| Feature | Type | Description |
|---|---|---|
| `user_id` | string | User identifier |
| `item_id` | string | Product identifier |

**User Demographics (8)**:
| Feature | Type | Description |
|---|---|---|
| `age` | int | User age |
| `gender` | binary | 0 = Female, 1 = Male |
| `user_level` | int | Platform user level (1-5) |
| `purchase_freq` | int | Historical purchase frequency |
| `total_spend` | float | Cumulative spending amount |
| `register_days` | int | Days since registration |
| `follow_num` | int | Number of users followed |
| `fans_num` | int | Number of followers |

**Seller-Controllable Product Features (6)**:
| Feature | Type | Description |
|---|---|---|
| `title_length` | int | Product title character count |
| `title_emo_score` | float | Title sentiment score (0-1) |
| `img_count` | int | Number of product images |
| `has_video` | binary | Whether listing includes video |
| `price` | float | Product price |
| `discount_rate` | float | Discount percentage (0-1) |

**Engagement Metrics (6)** - Identified as data leakage, dropped:
| Feature | Type | Description |
|---|---|---|
| `like_num` | int | Number of likes |
| `comment_num` | int | Number of comments |
| `share_num` | int | Number of shares |
| `collect_num` | int | Number of collections |
| `interaction_rate` | float | Composite interaction rate |
| `social_influence` | float | Social influence score |

**Behavioral Features (7)**:
| Feature | Type | Description |
|---|---|---|
| `is_follow_author` | binary | Whether user follows the seller |
| `add2cart` | binary | Whether product was added to cart |
| `coupon_received` | binary | Whether user received a coupon |
| `coupon_used` | binary | Whether coupon was used |
| `pv_count` | int | Page view count |
| `last_click_gap` | float | Days since last click |
| `purchase_intent` | float | Computed purchase intent score |
| `freshness_score` | float | Product listing freshness |

**Other**:
| Feature | Type | Description |
|---|---|---|
| `category` | string | Product category |
| `label` | binary | Target: 1 = purchased, 0 = not purchased |

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Data Leakage Analysis

A critical preprocessing step was identifying and removing **data leakage features**. Data leakage occurs when features that would not be available at prediction time are included in the training data, artificially inflating model performance.

**Leakage identification through causal reasoning:**

The following 6 engagement features were identified as **post-purchase artifacts**:

- `like_num`, `comment_num`, `share_num`, `collect_num`: These engagement metrics accumulate **after** users interact with and potentially purchase the product. In a production scenario, when a new product listing is created, these values are all zero.
- `interaction_rate`: Derived from the above engagement metrics.
- `social_influence`: Computed from social engagement data.

**Causal argument**: The temporal ordering is: *product listed -> user browses -> user purchases (or not) -> user engages (likes, comments, shares)*. Including post-purchase engagement as predictive features violates temporal causality and would make the model unusable for new product predictions.

These 6 features were dropped, reducing the feature set from 32 to 24 columns (after also dropping `user_id` and `item_id`).

### 3.2 Feature Engineering

**Category Encoding**: The `category` column (6 unique values) was encoded using `sklearn.preprocessing.LabelEncoder`:

| Category | Encoded Value |
|---|---|
| 其他 | 0 |
| 家居日用 | 1 |
| 数码家电 | 2 |
| 服饰鞋包 | 3 |
| 美妆个护 | 4 |
| 食品生鲜 | 5 |

**Feature Groups**: Features were organized into semantic groups for analysis:

- **Seller-controllable** (6): `title_length`, `title_emo_score`, `img_count`, `has_video`, `price`, `discount_rate`
- **User context** (8): `age`, `gender`, `user_level`, `purchase_freq`, `total_spend`, `register_days`, `follow_num`, `fans_num`
- **Behavioral** (7): `is_follow_author`, `add2cart`, `coupon_received`, `coupon_used`, `pv_count`, `last_click_gap`, `purchase_intent`, `freshness_score`
- **Category** (1): `category_encoded`
- **Target** (1): `label`

### 3.3 Train-Test Split

The dataset was split using stratified sampling to preserve the class distribution:

| Split | Size | Label Rate |
|---|---|---|
| Training | 80,000 (80%) | ~44.98% |
| Test | 20,000 (20%) | ~44.98% |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 3.4 Feature Scaling

`StandardScaler` was fitted on the training set and applied to both train and test sets. This is required for Logistic Regression but not for tree-based models (Random Forest, XGBoost), which are invariant to feature scaling.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 4. Model Construction & Comparison

### 4.1 Models Trained

Three classification models were trained and compared:

#### 4.1.1 Logistic Regression (LR)

- **Type**: Linear model with sigmoid activation
- **Configuration**: `max_iter=1000, random_state=42`
- **Input**: Scaled features (`X_train_scaled`)
- **Strengths**: Interpretable coefficients, fast training, provides probability calibration
- **Weaknesses**: Cannot capture non-linear feature interactions

#### 4.1.2 Random Forest (RF)

- **Type**: Ensemble of 200 decision trees with bagging
- **Configuration**: `n_estimators=200, max_depth=10, random_state=42, n_jobs=-1`
- **Input**: Unscaled features (`X_train`)
- **Strengths**: Handles non-linear relationships, robust to outliers, built-in feature importance
- **Weaknesses**: Slower inference, less interpretable than LR

#### 4.1.3 XGBoost (XGB)

- **Type**: Gradient-boosted decision trees
- **Configuration**: `n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'`
- **Input**: Unscaled features (`X_train`)
- **Strengths**: State-of-the-art performance on tabular data, handles missing values, regularization
- **Weaknesses**: More hyperparameters to tune, risk of overfitting

### 4.2 Evaluation Metrics

All models were evaluated on the held-out test set (20,000 samples) using five metrics:

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7127 | 0.7215 | 0.5883 | 0.6481 | 0.7584 |
| Random Forest | **0.7156** | **0.7266** | 0.5896 | 0.6510 | **0.7707** |
| XGBoost | 0.7133 | 0.7132 | **0.6065** | **0.6556** | 0.7697 |

**Key observations:**

1. **AUC-ROC**: Random Forest achieved the highest AUC (0.7707), closely followed by XGBoost (0.7697). This indicates RF has the best overall discrimination ability.
2. **Precision vs. Recall trade-off**: XGBoost has the highest recall (0.6065) and F1 (0.6556), meaning it catches more actual purchasers, while RF has the highest precision (0.7266), meaning fewer false positives.
3. **Accuracy**: All three models achieve similar accuracy (~71.3%), suggesting the task has inherent difficulty with the available features (after removing leakage features).
4. **LR as baseline**: Logistic Regression provides a competitive baseline, only ~1.2% below the tree-based models in AUC, suggesting the relationship is partially linear.

### 4.3 Model Selection

**Selected model: Random Forest** (AUC = 0.7707)

Selection criteria:
- Highest AUC-ROC, the primary metric for ranking binary classifiers
- Compatible with SHAP TreeExplainer for fast interpretability analysis
- Built-in feature importance for quick sanity checks
- Robust to hyperparameter choices

*[Placeholder: ROC curves comparison plot for all 3 models]*

*[Placeholder: Confusion matrices for all 3 models]*

---

## 5. Model Selection & Training Details

### 5.1 Training Pipeline

```
Raw Data (100K x 32)
    |
    v
Drop IDs + Leakage Features (100K x 24)
    |
    v
LabelEncoder(category) -> category_encoded (100K x 23 features + 1 target)
    |
    v
Stratified Train/Test Split (80/20)
    |
    v
StandardScaler (fit on train only)
    |
    +---> LR (on scaled data)
    +---> RF (on unscaled data)
    +---> XGBoost (on unscaled data)
    |
    v
Evaluate on Test Set (Accuracy, Precision, Recall, F1, AUC)
    |
    v
Select Best Model by AUC -> Random Forest
```

### 5.2 Hyperparameter Details

| Hyperparameter | LR | RF | XGBoost |
|---|---|---|---|
| n_estimators | - | 200 | 200 |
| max_depth | - | 10 | 6 |
| learning_rate | - | - | 0.1 |
| max_iter | 1000 | - | - |
| random_state | 42 | 42 | 42 |
| n_jobs | - | -1 (all cores) | - |
| eval_metric | - | - | logloss |

### 5.3 Feature Count

The final feature vector contains **23 features**:

```
['age', 'gender', 'user_level', 'purchase_freq', 'total_spend',
 'register_days', 'follow_num', 'fans_num', 'price', 'discount_rate',
 'title_length', 'title_emo_score', 'img_count', 'has_video',
 'is_follow_author', 'add2cart', 'coupon_received', 'coupon_used',
 'pv_count', 'last_click_gap', 'purchase_intent', 'freshness_score',
 'category_encoded']
```

---

## 6. Model Interpretability

### 6.1 Feature Importance (Tree-based)

Both Random Forest and XGBoost provide built-in feature importance scores (based on impurity reduction for RF, gain for XGBoost). Features were color-coded by their group:

- Green: Seller-controllable features
- Blue: User context features
- Orange: Behavioral features
- Gray: Category

*[Placeholder: Feature importance bar chart for RF and XGBoost side-by-side]*

Key finding: Among seller-controllable features, `price` and `discount_rate` have the highest importance, while `title_length` and `title_emo_score` have relatively low importance (~1-2%). This motivates the hybrid scoring approach discussed in Section 9.

### 6.2 SHAP Analysis

SHAP (SHapley Additive exPlanations) values were computed using `TreeExplainer` on a sample of 2,000 test instances to provide both global and local interpretability.

```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)
```

**Global SHAP summary** reveals:
- `purchase_intent` and `add2cart` are the top predictors (behavioral signals of imminent purchase)
- Among seller-controllable features, `price` has the highest SHAP impact
- User demographics (`age`, `gender`) have moderate but consistent effects

*[Placeholder: SHAP beeswarm plot]*

*[Placeholder: SHAP bar plot (mean |SHAP| values)]*

**Seller-controllable SHAP analysis**: A focused SHAP analysis on only the 6 seller-controllable features helps sellers understand which product attributes they can change to improve conversion.

*[Placeholder: SHAP beeswarm plot for seller-controllable features only]*

### 6.3 Partial Dependence Plots (PDP)

PDPs show the marginal effect of each seller-controllable feature on the predicted purchase probability, averaging over all other features.

```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_test, features=controllable_idx)
```

*[Placeholder: PDP grid for 6 seller-controllable features]*

Key insights from PDPs:
- **Price**: Negative monotonic relationship -- lower prices consistently increase purchase probability
- **Discount rate**: Moderate positive effect up to ~20%, then diminishing returns
- **Image count**: Slight positive effect plateauing around 3-5 images
- **Has video**: Binary jump -- adding video provides a discrete lift
- **Title length**: Nearly flat, confirming low ML importance
- **Sentiment**: Weak positive trend

### 6.4 Counterfactual Analysis

For each seller-controllable feature, we simulate "what if" scenarios: what would happen to the purchase probability if a specific feature were changed to the category benchmark (median of purchased items)?

```python
def simulate_counterfactual(model, X_row, feature_name, new_value, feature_cols):
    X_original = X_row.values.reshape(1, -1)
    original_prob = model.predict_proba(X_original)[0, 1]
    X_modified = X_original.copy()
    X_modified[feature_name] = new_value
    new_prob = model.predict_proba(X_modified)[0, 1]
    return original_prob, new_prob, new_prob - original_prob
```

**Category benchmarks** are computed from the median feature values of **purchased items only** within each category:

| Category | title_length | title_emo_score | img_count | has_video | price | discount_rate |
|---|---|---|---|---|---|---|
| 服饰鞋包 | 27 | 0.604 | 3 | 0 | 73.25 | 9.3% |
| 数码家电 | 27 | 0.609 | 3 | 0 | 73.85 | 9.1% |
| 食品生鲜 | 28 | 0.607 | 3 | 0 | 73.15 | 9.1% |
| 美妆个护 | 27 | 0.606 | 3 | 0 | 75.04 | 8.5% |
| 家居日用 | 28 | 0.607 | 3 | 0 | 72.48 | 8.6% |
| 其他 | 28 | 0.609 | 3 | 0 | 74.53 | 8.6% |

*[Placeholder: Counterfactual analysis bar chart showing expected probability delta for each feature]*

---

## 7. Sentiment Analysis Pipeline

### 7.1 Problem Statement

The dataset contains a `title_emo_score` feature representing the emotional sentiment of product titles. To enable real-time predictions on new product titles in the dashboard, we need a sentiment analysis model that can compute this score from raw Chinese text.

### 7.2 SnowNLP Integration

We use **SnowNLP**, a Python library for Chinese text processing, to compute sentiment scores from product titles:

```python
from snownlp import SnowNLP
raw_score = SnowNLP(title_text).sentiments  # Returns 0-1
```

### 7.3 Distribution Mismatch Problem

A critical issue was discovered: SnowNLP's output distribution differs significantly from the dataset's `title_emo_score` distribution.

| Property | SnowNLP Output | Dataset `title_emo_score` |
|---|---|---|
| Distribution Shape | U-shaped (bimodal) | Bell-shaped (unimodal) |
| Standard Deviation | ~0.33 | ~0.15 |
| Range | 0.0002 - 0.9999 | 0.059 - 0.976 |
| Mean | ~0.55 | ~0.60 |

This distribution shift means directly using SnowNLP scores as model input would produce unreliable predictions, as the model was trained on a different distribution.

### 7.4 Beta Quantile Calibration

To align SnowNLP outputs with the training data distribution, we implemented **Beta distribution quantile mapping**:

**Step 1: Fit Beta distributions to both distributions**

- Source (SnowNLP): Beta(alpha=0.6211, beta=0.5600) -- captures the U-shape
- Target (dataset): Beta(alpha=6.0003, beta=4.0162) -- captures the bell-shape

**Step 2: Quantile mapping transformation**

For a raw SnowNLP score `s`:

```
percentile = Beta_source.cdf(s)          # Map to uniform [0,1]
calibrated = Beta_target.ppf(percentile)  # Map back to target distribution
```

```python
def calibrate_sentiment(raw_score):
    clipped = np.clip(raw_score, 1e-6, 1 - 1e-6)
    percentile = stats.beta.cdf(clipped, source_alpha, source_beta)
    calibrated = stats.beta.ppf(percentile, target_alpha, target_beta)
    return float(calibrated)
```

**Why Beta quantile mapping?**

1. It preserves the **percentile rank** of each score -- a score at the 80th percentile in SnowNLP's distribution maps to the 80th percentile in the dataset's distribution.
2. The Beta distribution is the natural choice for modeling values bounded in [0,1] with various shapes.
3. It handles the bimodal-to-unimodal transformation gracefully.

**Limitation**: This calibration assumes that the percentile rank of a sentiment score is meaningful across distributions, which may not hold perfectly for edge cases. The calibrated score should be interpreted as "a score consistent with the training data distribution" rather than a ground-truth sentiment value.

---

## 8. User Clustering

### 8.1 Motivation

Different user segments respond differently to product listings. A product that converts well for one segment may not work for another. User clustering enables:
1. **Per-cluster prediction**: Instead of using a single "average user" for prediction, we predict for each cluster separately and compute a weighted average.
2. **Targeted marketing**: Sellers can tailor their strategies to the most profitable user segment.

### 8.2 Feature Selection for Clustering

After testing multiple feature combinations and evaluating with Silhouette score (cluster separation quality) and purchase rate variance (business relevance), the following 7 features were selected:

| Feature | Rationale |
|---|---|
| `age` | Core demographic |
| `gender` | Core demographic |
| `user_level` | Platform engagement indicator |
| `purchase_freq` | Purchase behavior frequency |
| `total_spend` | Spending power |
| `purchase_intent` | Behavioral intent signal |
| `add2cart` | Cart behavior (strong purchase signal) |

Features were standardized using `StandardScaler` before clustering.

### 8.3 Clustering Method

**Algorithm**: K-Means with k=4 (selected via elbow method and Silhouette analysis)

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['user_cluster'] = kmeans.fit_predict(X_user_scaled)
```

**Cluster quality metrics:**
- Silhouette Score: 0.24
- Purchase Rate Variance across clusters: 0.42

### 8.4 Cluster Profiles

| Cluster | Name | Size | % | Purchase Rate | Key Characteristics |
|---|---|---|---|---|---|
| C0 | Female / Frequent / Low-intent | 8,995 | 9.0% | **41.0%** | High purchase frequency (36), female-dominated (70%), moderate intent |
| C1 | High-intent / Cart-active | 22,338 | 22.3% | **77.0%** | Highest purchase intent (7.2), almost all added to cart (99%), mixed gender |
| C2 | Female / Low-intent | 43,210 | 43.2% | **35.1%** | Largest segment, all female, low intent (1.4), no cart additions |
| C3 | Male / Low-intent | 25,457 | 25.5% | **35.0%** | All male, low intent (1.4), no cart additions |

**Key insight**: Cluster C1 (High-intent / Cart-active) has a dramatically higher purchase rate (77%) compared to other clusters (35-41%). This cluster is defined primarily by high `purchase_intent` and `add2cart` signals, confirming that behavioral intent is the strongest predictor of conversion.

### 8.5 Per-Cluster Weighted Prediction

Instead of using a global user median for prediction, the system predicts separately for each cluster using cluster-specific user medians, then computes a weighted average:

```python
def predict_cluster_weighted(product_params, feat_cols, cat_map, coupon=0):
    per_cluster = {}
    weighted_sum = 0.0
    for c_id in range(OPTIMAL_K):
        user_params = cluster_medians[c_id]  # Cluster-specific median
        fv = build_feature_vector(product_params, user_params, feat_cols, cat_map)
        prob = model.predict_proba(fv)[0, 1]
        w = cluster_weights[c_id]  # Proportion of users in cluster
        weighted_sum += prob * w
        per_cluster[c_id] = {'prob': prob, 'weight': w}
    return weighted_sum, per_cluster
```

This approach respects user heterogeneity: the same product may have a 62.7% purchase probability for high-intent users (C1) but only 17.8% for low-intent users (C2/C3).

*[Placeholder: Radar chart showing normalized cluster profiles]*

*[Placeholder: Marketing effectiveness chart (coupon lift and video lift by cluster)]*

---

## 9. Recommendation Engine

### 9.1 Hybrid Scoring System

A key finding from interpretability analysis was that the ML model assigns low importance (~1-2%) to title-related features (`title_length`, `title_emo_score`). However, domain knowledge tells us that title quality is crucial for e-commerce conversion. Similarly, spam titles (e.g., repeated characters) are clearly bad but may not be penalized by the ML model.

To address this, we implemented a **hybrid scoring system** combining:

1. **ML Score** (50% weight): Normalized purchase probability from the Random Forest model
2. **Domain Knowledge Score** (50% weight): Rules-based scoring on 7 dimensions

### 9.2 Domain Knowledge Scoring

The domain score evaluates products on 7 weighted dimensions:

| Dimension | Weight | Scoring Logic |
|---|---|---|
| Title Quality | 25% | Composite of char quality, category relevance, info density (see 9.3) |
| Title Length | 10% | Optimal within +/-5 chars of category benchmark |
| Sentiment | 10% | Higher positive sentiment -> higher score, full marks >= 0.7 |
| Images | 15% | 3-6 images optimal (score=1.0), 0 images (score=0.2) |
| Video | 15% | Has video = 1.0, no video = 0.4 |
| Price | 15% | At or below category benchmark = high score |
| Discount | 10% | 5-20% discount is optimal |

### 9.3 Title Quality Assessment

Title quality is evaluated on 3 sub-dimensions to detect spam and low-quality titles:

**1. Character Quality (weight: 35%)** -- Anti-spam detection:
- Unique character ratio: Low ratio (< 0.3) indicates repetitive content, score multiplied by 0.15
- Consecutive repeated characters: 4+ consecutive repeats -> score * 0.2
- Dominant character ratio: Single character > 50% of title -> score * 0.2

Example: "春季女装啊啊啊啊" (spam with repeated "啊") -> char_quality score ~0.036

**2. Category Relevance (weight: 40%)** -- Keyword matching:
- A curated dictionary of ~50-60 keywords per category is maintained
- Score = min(1.0, matched_keywords / 3)

Example: "春季时尚女装连衣裙" matched "女装", "连衣裙", "时尚" in 服饰鞋包 -> relevance = 1.0

**3. Information Density (weight: 25%)** -- Content richness:
- Effective unique characters (after deduplication) / total characters
- Measures how much information each character conveys

### 9.4 Counterfactual-Based Recommendations

For each seller-controllable feature, the system computes the expected purchase rate lift if the feature is changed to the category benchmark:

```python
for feat in SELLER_CONTROLLABLE:
    current_val = product_params[feat]
    ideal_val = category_benchmark[feat]
    original_prob, new_prob, delta = simulate_counterfactual(
        model, feature_vector, feat, ideal_val, feature_cols
    )
    # delta = new_prob - original_prob (positive means improvement)
```

Recommendations are generated based on:
- Features below category benchmark -> suggest increase with expected lift
- Features above category benchmark -> suggest alignment
- Title quality issues -> specific suggestions (add keywords, reduce repetition)
- Missing video -> suggest adding with expected lift
- Price above median -> suggest reduction with expected lift

---

## 10. System Architecture & Pipeline

### 10.1 System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline                         │
│                                                         │
│  social_ecommerce_data.csv (100K x 32)                 │
│         │                                               │
│         v                                               │
│  analysis.ipynb                                         │
│    ├── Data Leakage Removal (drop 6 engagement feats)  │
│    ├── EDA & Visualization                              │
│    ├── Feature Engineering (LabelEncoder, scaling)      │
│    ├── Multi-Model Training (LR, RF, XGBoost)          │
│    ├── Model Interpretability (SHAP, PDP, CF)          │
│    ├── K-Means Clustering (4 clusters)                 │
│    └── Save Artifacts                                   │
│         │                                               │
│         v                                               │
│  model_artifacts.pkl (19MB)                             │
│    ├── Random Forest model                              │
│    ├── Feature columns & scalers                        │
│    ├── Category benchmarks (per-cat medians)            │
│    ├── KMeans model & cluster profiles                  │
│    ├── Cluster medians & weights                        │
│    ├── Sentiment calibration params                     │
│    └── Model comparison results                         │
│         │                                               │
│         v                                               │
│  processed_data.csv (9MB, 100K x 26)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                    │
                    v
┌─────────────────────────────────────────────────────────┐
│               Streamlit Dashboard (app.py)               │
│                                                         │
│  Page 1: Product Analyzer                               │
│    ├── Title Input -> SnowNLP -> Beta Calibration      │
│    ├── Product Feature Input (category, price, etc.)   │
│    ├── Per-Cluster Weighted Prediction                  │
│    ├── Hybrid Score (ML 50% + Domain 50%)              │
│    ├── Domain Score Breakdown (7 dimensions)           │
│    ├── Title Quality Detail (3 sub-dimensions)         │
│    ├── Counterfactual Analysis                          │
│    ├── Target User Persona                              │
│    └── Actionable Seller Recommendations                │
│                                                         │
│  Page 2: User Clusters                                  │
│    ├── Cluster Overview Cards                           │
│    ├── Feature Radar Chart                              │
│    ├── Marketing Effectiveness (Coupon/Video Lift)     │
│    ├── Detailed Cluster Profiles Table                  │
│    └── Purchase Rate by Category x Cluster             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 10.2 File Structure

```
Final Project/
├── analysis.ipynb          # Full ML pipeline (45 cells)
├── app.py                  # Streamlit dashboard (2 pages)
├── model_artifacts.pkl     # Serialized models & artifacts (19MB)
├── processed_data.csv      # Processed dataset (9MB, 100K x 26)
├── social_ecommerce_data.csv  # Raw dataset (13MB, 100K x 32)
├── requirements.txt        # Python dependencies
└── report.md               # This report
```

### 10.3 Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | >= 2.0 | Data manipulation |
| numpy | >= 1.24 | Numerical computation |
| scikit-learn | >= 1.3 | ML models, preprocessing, evaluation |
| xgboost | >= 2.0 | Gradient boosted trees |
| shap | >= 0.43 | Model interpretability |
| matplotlib | >= 3.7 | Static visualization |
| seaborn | >= 0.12 | Statistical visualization |
| plotly | >= 5.15 | Interactive visualization |
| streamlit | >= 1.30 | Dashboard framework |
| snownlp | latest | Chinese text sentiment analysis |
| scipy | latest | Beta distribution for sentiment calibration |

### 10.4 Prediction Pipeline (Runtime)

When a user inputs a product in the dashboard, the following pipeline executes:

```
User Input (title, category, price, discount, images, video, coupon)
    │
    ├── SnowNLP(title) -> raw_sentiment
    │       │
    │       v
    │   Beta Quantile Calibration -> calibrated_sentiment
    │
    ├── For each cluster c in {0, 1, 2, 3}:
    │       │
    │       v
    │   Build feature vector (product_params + cluster_median_user_params)
    │       │
    │       v
    │   Random Forest predict_proba -> cluster_prob[c]
    │
    ├── weighted_avg = sum(cluster_prob[c] * cluster_weight[c])
    │
    ├── Title Quality Assessment (char_quality + relevance + density)
    │
    ├── Domain Score (7 weighted dimensions)
    │
    ├── Hybrid Score = 0.5 * ML_score + 0.5 * Domain_score
    │
    ├── Counterfactual Analysis (6 seller features vs benchmarks)
    │
    └── Generate Recommendations
```

---

## 11. Dashboard & Result Presentation

### 11.1 Page 1: Product Analyzer

The Product Analyzer is a consolidated page that provides sellers with comprehensive analysis of their product listing.

**Input Section:**
- Product title (Chinese text, auto-analyzed for sentiment and quality)
- Category selector (6 categories)
- Product features: image count, price, discount rate, video presence, coupon option

**Output Sections:**

1. **Prediction Results**: Two gauges showing cluster-weighted purchase probability and hybrid score. A bar chart shows per-cluster purchase probability breakdown.

2. **Domain Score Breakdown**: Horizontal bar chart showing scores across 7 dimensions (title quality, length, sentiment, images, video, price, discount) with color coding (green >= 70, orange >= 40, red < 40). Title quality sub-breakdown shows character quality, category relevance, and information density.

3. **Counterfactual Analysis**: Horizontal bar chart showing expected purchase rate change if each seller-controllable feature is changed to the category benchmark value.

4. **Target User Persona**: Identifies the best user cluster for the product's category, showing demographics and purchase rate. Bar chart compares purchase rates across all clusters for the selected category.

5. **Seller Recommendations**: Actionable recommendation cards with specific suggestions and expected lift values. Each recommendation includes the current value, benchmark value, and the counterfactual expected improvement.

6. **Category Benchmark Reference**: Table comparing the product's current feature values against category benchmarks.

*[Placeholder: Product Analyzer page screenshot]*

### 11.2 Page 2: User Clusters

The User Clusters page provides insights into user segmentation.

**Sections:**

1. **Cluster Overview Cards**: 4 cards showing each cluster's purchase rate, name, user count, and percentage.

2. **Feature Radar Chart**: Normalized radar plot comparing cluster profiles across 6 dimensions (age, user_level, purchase_freq, total_spend, purchase_intent, add2cart).

3. **Marketing Effectiveness**: Grouped bar chart showing coupon lift and video lift for each cluster, helping sellers identify which marketing actions work best for which segments.

4. **Detailed Cluster Profiles**: Table with all cluster attributes (age, gender, user_level, purchase_freq, total_spend, purchase_intent, add2cart, purchase_rate, count).

5. **Category x Cluster Cross-Analysis**: Grouped bar chart showing purchase rate for every category-cluster combination, helping sellers identify the most responsive segments for their product category.

*[Placeholder: User Clusters page screenshot]*

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

1. **Data leakage trade-off**: Removing 6 engagement features reduced AUC from potentially inflated values to ~0.77. While this reflects realistic prediction capability, the accuracy is moderate. The remaining behavioral features (`add2cart`, `purchase_intent`) are still strong signals that may not always be available at listing time.

2. **Sentiment calibration approximation**: The Beta quantile mapping assumes percentile preservation between SnowNLP and dataset distributions. This is an approximation that may not hold perfectly, especially for extreme values.

3. **Static benchmarks**: Category benchmarks are computed once from training data and do not update as market conditions change.

4. **Clustering granularity**: K-Means with k=4 provides a coarse segmentation. The Silhouette score of 0.24 indicates moderate cluster quality, with potential for improvement.

5. **Title quality heuristics**: The keyword-based category relevance scoring is limited to a manually curated dictionary and may miss new or niche product terms.

6. **No temporal modeling**: The current model does not account for time-series effects (seasonality, trends, product lifecycle).

### 12.2 Future Improvements

1. **Feature enrichment**: Incorporate product description text features, image quality scores, and seller reputation metrics.
2. **Advanced NLP**: Replace SnowNLP with transformer-based Chinese sentiment models (e.g., BERT-based) for more accurate sentiment analysis.
3. **Dynamic benchmarks**: Implement rolling benchmarks that update with recent data.
4. **A/B testing framework**: Integrate with live A/B testing to validate recommendations against actual conversion data.
5. **Deep learning models**: Experiment with neural network architectures (TabNet, DeepFM) for potentially better prediction accuracy.
6. **Real-time clustering**: Implement online clustering that updates user segments as new data arrives.

---

## Appendix

### A. Reproducibility

To reproduce the results:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis notebook
jupyter notebook analysis.ipynb
# Execute all cells to generate model_artifacts.pkl and processed_data.csv

# Launch the dashboard
streamlit run app.py
```

### B. Model Artifacts Structure

The `model_artifacts.pkl` file contains the following serialized objects:

| Key | Type | Description |
|---|---|---|
| `model` | RandomForestClassifier | Trained purchase prediction model |
| `model_name` | str | "Random Forest" |
| `feature_cols` | list[str] | 23 feature column names |
| `scaler` | StandardScaler | Feature scaler (for LR compatibility) |
| `label_encoder` | LabelEncoder | Category encoder |
| `category_mapping` | dict | Category -> encoded value mapping |
| `benchmarks` | dict | Per-category feature benchmarks from purchased items |
| `comparison_df` | DataFrame | Model comparison metrics (3 models x 5 metrics) |
| `kmeans` | KMeans | Trained clustering model (k=4) |
| `scaler_kmeans` | StandardScaler | Scaler for clustering features |
| `user_features_for_cluster` | list[str] | 7 features used for clustering |
| `cluster_profiles` | DataFrame | Mean feature values per cluster |
| `cluster_names` | dict | Human-readable cluster names |
| `cluster_medians` | dict | Per-cluster user feature medians (for prediction) |
| `cluster_weights` | dict | Cluster size proportions |
| `OPTIMAL_K` | int | 4 |
| `SELLER_CONTROLLABLE` | list[str] | 6 seller-controllable feature names |
| `pop_stats` | dict | Per-category engagement statistics |
| `sentiment_calibration` | dict | Beta distribution parameters for sentiment calibration |
