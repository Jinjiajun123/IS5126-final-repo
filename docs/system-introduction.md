# NexusAnalytics -- Merchant AI Suite: System Features Overview

---

## 1. System Overview

NexusAnalytics (Merchant AI Suite) is an AI-powered analytics platform for e-commerce merchants. It focuses on two core scenarios -- **product optimization** and **precision operations** -- providing end-to-end support from product diagnostics and AI creative generation to user intent analysis.

Core capabilities include:

- **ML Prediction Engine**: Machine learning models trained on real e-commerce data to predict purchase conversion probability, supporting both listing-time feature prediction and session-time behavioral prediction
- **LLM Deep Analysis**: Integrates the Doubao **doubao-seed-pro** large language model, combining ML quantitative predictions with product information to generate expert-level optimization recommendations and target persona analysis
- **AI Creative Generation**: Integrates the Doubao **doubao-seedream-5.0** text-to-image model, automatically generating high-quality advertising visuals based on product descriptions and style preferences
- **User Behavior Simulation & Intent Segmentation**: Simulates session behavior data based on 4 user archetypes, predicts purchase intent via ML models, segments users by probability thresholds, and outputs targeted operational strategies

The platform uses a fixed left-side navigation bar with a main content area on the right, consisting of three core modules:

| Navigation | Description |
| --- | --- |
| **Analysis** | Dual-engine product diagnostics (ML + LLM), delivering unified scores and optimization reports |
| **Ad Studio** | AI-powered advertising creative generation studio |
| **User Insights** | Session-based user intent analysis with actionable operational recommendations |

> [截图：平台整体界面，展示左侧导航栏和右侧主内容区]

---

## 2. Analysis -- Analytics Hub (Product Diagnostics)

### 2.1 Overview

Analysis serves as the platform's homepage and is positioned as a **comprehensive pre-listing diagnostic tool**. After merchants input product parameters and upload promotional images, the system simultaneously launches the ML prediction engine and the doubao-seed-pro LLM analysis engine ("Dual Engine"), producing a complete diagnostic report with unified scores, strengths/weaknesses analysis, and target persona profiling.

> [截图：Analytics Hub 完整页面，左侧输入表单 + 右侧手机模拟预览]

### 2.2 Product Parameter Input

The left-side input panel allows merchants to enter product details:

| Field | Description |
| --- | --- |
| Product Title | Product title; the system automatically analyzes title length and sentiment, which factor into search ranking evaluation |
| Category | Product category (Clothing / Electronics / Food / Beauty / Home / Others) |
| Price | Product selling price |
| Discount Rate | Discount rate (0-1, e.g. 0.1 = 10% off) |
| Coupon Amount | Coupon toggle + amount input; Toggle switch controls activation |
| Hero Image | Product image upload, supporting up to 10 images |

> [截图：商品参数输入面板填写状态]

### 2.3 Real-Time Mobile Preview

Before triggering analysis, the right side displays a **phone simulator** that provides a real-time preview of how the product would appear in the Taobao/Tmall App. This includes the product image carousel, price bar, title and service tags, coupon redemption bar, category labels, and bottom action bar (Store / Chat / Star / Add to Cart / Buy Now). Merchants can visually assess the product page presentation before submitting for analysis.

> [截图：手机模拟器预览效果]

### 2.4 Dual-Engine Analysis

Clicking **"Run Analytics"** launches both analysis engines simultaneously:

- **ML Engine**: Constructs a feature vector from product parameters, uses the calibrated ML model to predict purchase probability for each user cluster, and computes a weighted aggregate probability. It also calculates a domain knowledge score by benchmarking against category averages, then fuses both into a unified score (unified_score).
- **LLM Engine (doubao-seed-pro)**: Assembles product information, ML predictions, and user persona data into a structured prompt for the large language model. If images are uploaded, they are included for visual analysis; if not, the prompt explicitly flags the absence of promotional images as a key weakness.

### 2.5 Diagnostic Report

Upon completion, the right side displays a diagnostic report card with four core sections:

**Unified Score**: A combined score (out of 100) fusing ML probability and LLM evaluation. Grade tiers: Excellent (>= 80) / Good (60-79) / Average (30-59) / Needs Work (< 30).

**Conversion Drivers (Strengths)**: Marked with a green left border, listing the product's key advantages with supporting data and category benchmarks -- such as keyword relevance, image quality, and price competitiveness.

**Weaknesses**: Marked with an orange left border, listing areas for improvement with specific suggestions -- such as incomplete title information (missing color, brand, fabric attributes), insufficient image count, or lack of coupons. Each item includes expected improvement metrics.

**Target Persona Analysis**: Marked with a purple left border, describing the user segment most likely to convert (gender, age range, spending level, behavioral preferences) along with tailored operational recommendations.

> [截图：完整诊断报告，展示评分 + Strengths + Weaknesses + Persona]

A **"Use AI Generator ->"** link at the bottom of the report allows one-click navigation to Ad Studio for generating optimized advertising creatives.

---

## 3. Ad Studio -- AI Creative Generation Studio

### 3.1 Overview

Ad Studio is an **AI-powered advertising creative generation tool** built on the Doubao **doubao-seedream-5.0** text-to-image model. Merchants input product descriptions and visual style preferences to instantly generate professional-grade advertising visuals.

> [截图：Ad Studio 完整页面]

### 3.2 Campaign Brief Input

| Field | Description |
| --- | --- |
| Item Description | Product description copy; more detailed descriptions yield better results |
| Target Audience | Target audience profile, e.g. "18-25 young females", "25-35 Urban" |
| Aesthetic Preset | Visual style preset: Clean & Minimalist / Earthy Lifestyle / Dark High-End / Custom Style |
| Custom Style Keywords | Appears when Custom Style is selected; enter custom style keywords |
| Original Product Photo | Upload an original product photo as a reference for AI generation |

### 3.3 Creative Generation & Gallery

Clicking **"Generate Images"** triggers the doubao-seedream-5.0 model to generate advertising images, displayed in a **2x2 grid gallery** with a 4:5 portrait aspect ratio. Hovering over an image slightly enlarges it and reveals a **"Save Asset"** download button.

> [截图：生成结果 Gallery]

---

## 4. User Insights -- User Intent Analysis & Operational Recommendations

### 4.1 Overview

User Insights is a **session-based user segmentation and precision operations decision tool**. After selecting a listed product, the system collects user session behavior data, uses the ML model to predict each user's purchase probability, segments users into three intent tiers (High / Medium / Low), and generates actionable recommendations for each tier.

> [截图：User Insights 完整页面]

### 4.2 Product Selection & Summary

The left panel provides category filtering and product selection. Once a product is selected, a summary card displays basic attributes (category, price, discount, image count), engagement metrics (likes, favorites, comments, shares), and purchase conversion rate.

> [截图：商品选择面板及摘要信息]

### 4.3 User Behavior Simulation & Intent Segmentation

Clicking **"Analyze Users"** triggers the system to collect users who have interacted with the product from real data, and simulate session behavior based on 4 user archetypes (Loyal 15% / Active 30% / Casual 35% / New 20%). Each user generates 8 behavioral features (follow status, add-to-cart, coupon received/used, page views, click gap, purchase intent, freshness).

The ML model predicts each user's purchase probability, then segments them into three intent tiers by threshold:

| Tier | Probability Range | Description |
| --- | --- | --- |
| **High Intent** | >= 55% | Strong purchase signals; likely to convert organically |
| **Medium Intent** | 35% - 55% | On the fence; highest ROI for marketing interventions |
| **Low Intent** | < 35% | Casual browsers; require significant incentives to convert |

### 4.4 Analysis Results

#### Overview Stats Bar

A four-cell horizontal bar at the top displays key metrics: total users analyzed, average purchase probability, high-intent user count (highlighted), and high-intent user percentage.

> [截图：Overall Stats Bar]

#### Purchase Probability Distribution Chart

An ECharts bar chart visualizing the probability distribution across all users. Bars are color-coded by intent tier (green / amber / red), with dashed lines marking tier boundaries. A summary at the bottom shows user counts and percentages for each tier.

> [截图：概率分布直方图]

#### Four Actionable Recommendation Cards

The core output of the analysis, providing targeted operational strategies for different user segments:

**1. Coupon Targeting** ("Best ROI" tag)
Identifies Medium Intent users who have not yet received coupons -- these users are on the fence and most responsive to incentives. Displays target user count and estimated lift in percentage points, with a **"Send Coupons"** one-click action button.

**2. High-Intent Coupon Decision**
Compares probability differences between coupon-received and non-coupon groups among high-intent users. If the lift is less than 2pp, the system recommends "Save Margin" (skip coupons to preserve profit). If the lift is significant, it recommends "Send Coupon" to accelerate conversion.

**3. Traffic Push**
Identifies users with high purchase probability but low browsing depth (pv_count <= 3) -- they show purchase intent but haven't fully engaged with the product, making them ideal candidates for traffic investment. Displays candidate count and average probability, with a **"Start Traffic Push"** one-click action button.

**4. Priority Action -- Low-Intent Uplift**
A red-highlighted card that analyzes behavioral bottlenecks among low-intent users and dynamically generates improvement suggestions. Examples: low add-to-cart rate prompts CTA optimization; low follow rate prompts follow incentives; shallow browsing depth prompts first-screen content improvement; declining activity prompts time-limited promotions.

> [截图：四类运营建议卡片]

#### High-Intent User Ranking Table

A data table displaying the Top 50 highest-probability users, including User ID, Intent Tier (color-coded badge), Purchase Probability (visual progress bar), Age, Gender, Spend, Cart status, and Page Views -- enabling merchants to target high-value users directly.

> [截图：高意图用户排行表]
