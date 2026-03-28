import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from snownlp import SnowNLP
from scipy import stats
import re
from collections import Counter

# ─── Page Config ───
st.set_page_config(
    page_title="Seller Recommendation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .metric-card p { margin: 0; }
    .rec-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 1rem 1.2rem; margin: 0.6rem 0; border-radius: 0 8px 8px 0;
    }
    .rec-card-positive { border-left-color: #00c853; }
    .rec-card-negative { border-left-color: #ff5252; }
    .rec-card h4 { margin: 0 0 0.3rem 0; color: #333; }
    .rec-card p { margin: 0.2rem 0; color: #666; font-size: 0.9rem; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #333;
        border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .title-card {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        border: 1px solid #9fa8da;
    }
    .title-card h4 { margin: 0 0 0.5rem 0; color: #283593; }
    .title-card .score { font-size: 1.5rem; font-weight: bold; color: #1a237e; }
    .pop-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        border: 1px solid #ffcc80;
    }
    .pop-card h4 { margin: 0 0 0.3rem 0; color: #e65100; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ───
@st.cache_resource
def load_artifacts():
    with open('model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('processed_data.csv')

@st.cache_data
def load_raw_data():
    return pd.read_csv('social_ecommerce_data.csv')

try:
    artifacts = load_artifacts()
    df = load_data()
    df_raw = load_raw_data()
except FileNotFoundError:
    st.error("Please run analysis.ipynb first to generate model_artifacts.pkl and processed_data.csv")
    st.stop()

model = artifacts['model']
feature_cols = artifacts['feature_cols']
benchmarks = artifacts['benchmarks']
cluster_names = artifacts['cluster_names']
cluster_profiles = artifacts['cluster_profiles']
SELLER_CONTROLLABLE = artifacts['SELLER_CONTROLLABLE']
comparison_df = artifacts['comparison_df']
category_mapping = artifacts['category_mapping']
OPTIMAL_K = artifacts['OPTIMAL_K']
pop_stats = artifacts.get('pop_stats', {})
sent_cal = artifacts.get('sentiment_calibration', {})
cluster_medians = artifacts.get('cluster_medians', {})
cluster_weights = artifacts.get('cluster_weights', {})


# ─── Helper Functions ───
def calibrate_sentiment(raw_score):
    """Calibrate SnowNLP score (U-shaped) to match dataset distribution (bell-shaped)
    using Beta CDF quantile mapping."""
    if not sent_cal:
        return raw_score
    clipped = np.clip(raw_score, 1e-6, 1 - 1e-6)
    percentile = stats.beta.cdf(clipped, sent_cal['source_alpha'], sent_cal['source_beta'])
    calibrated = stats.beta.ppf(percentile, sent_cal['target_alpha'], sent_cal['target_beta'])
    return float(calibrated)


def compute_sentiment(text):
    """Compute sentiment score using SnowNLP, returns (raw, calibrated)."""
    if not text or not text.strip():
        return 0.5, 0.5
    try:
        raw = SnowNLP(text).sentiments
        cal = calibrate_sentiment(raw)
        return raw, cal
    except Exception:
        return 0.5, 0.5


def simulate_counterfactual(mdl, X_row_values, feature_name, new_value, feat_cols):
    vals = X_row_values.values if hasattr(X_row_values, 'values') else list(X_row_values)
    X_original = pd.DataFrame([vals], columns=feat_cols)
    original_prob = mdl.predict_proba(X_original)[0, 1]
    X_modified = X_original.copy()
    X_modified[feature_name] = new_value
    new_prob = mdl.predict_proba(X_modified)[0, 1]
    return original_prob, new_prob, new_prob - original_prob


def build_feature_vector(product_params, user_params, feat_cols, cat_map):
    feature_dict = {}
    feature_dict.update(product_params)
    feature_dict.update(user_params)
    vec = []
    for col in feat_cols:
        if col == 'category_encoded':
            vec.append(cat_map.get(product_params.get('category', ''), 0))
        elif col in feature_dict:
            vec.append(feature_dict[col])
        else:
            vec.append(0)
    return vec


def predict_proba_safe(mdl, feature_vec, feat_cols):
    X = pd.DataFrame([feature_vec], columns=feat_cols)
    return mdl.predict_proba(X)[0, 1]


def get_default_user_params():
    return {
        'age': int(df['age'].median()),
        'gender': int(df['gender'].mode().iloc[0]),
        'user_level': int(df['user_level'].median()),
        'purchase_freq': int(df['purchase_freq'].median()),
        'total_spend': float(df['total_spend'].median()),
        'register_days': int(df['register_days'].median()),
        'follow_num': int(df['follow_num'].median()),
        'fans_num': int(df['fans_num'].median()),
        'is_follow_author': 0,
        'add2cart': 0,
        'coupon_received': 0,
        'coupon_used': 0,
        'pv_count': int(df['pv_count'].median()),
        'last_click_gap': float(df['last_click_gap'].median()),
        'purchase_intent': float(df['purchase_intent'].median()),
        'freshness_score': float(df['freshness_score'].median()),
    }


def get_cluster_user_params(cluster_id):
    if cluster_id in cluster_medians:
        return dict(cluster_medians[cluster_id])
    return get_default_user_params()


def predict_cluster_weighted(product_params, feat_cols, cat_map, coupon=0):
    """Predict purchase probability using per-cluster user medians, weighted by cluster size."""
    per_cluster = {}
    weighted_sum = 0.0
    for c_id in range(OPTIMAL_K):
        user_params = get_cluster_user_params(c_id)
        user_params['coupon_received'] = coupon
        fv = build_feature_vector(product_params, user_params, feat_cols, cat_map)
        prob = predict_proba_safe(model, fv, feat_cols)
        w = cluster_weights.get(c_id, 1.0 / OPTIMAL_K)
        weighted_sum += prob * w
        per_cluster[c_id] = {
            'prob': prob,
            'weight': w,
            'name': cluster_names.get(c_id, f'Cluster {c_id}'),
        }
    return weighted_sum, per_cluster


# ─── Title Quality Scoring ───
_CATEGORY_KEYWORDS = {
    '服饰鞋包': ['女装','男装','连衣裙','T恤','衬衫','外套','裤','裙','鞋','包','帽','袜',
                '上衣','卫衣','毛衣','羽绒','大衣','风衣','夹克','西装','牛仔','短袖','长袖',
                '运动鞋','高跟','靴','凉鞋','拖鞋','背包','手提','钱包','皮带','围巾',
                '时尚','百搭','修身','显瘦','宽松','新款','韩版','潮流','休闲','正装',
                '春','夏','秋','冬','棉','麻','丝','绸','皮','革','纯棉','真丝','雪纺',
                '码','均码','加大','弹力','透气','保暖','防风','防水'],
    '数码家电': ['手机','电脑','笔记本','平板','耳机','音箱','键盘','鼠标','显示器','摄像',
                '充电','电池','数据线','保护壳','贴膜','支架','适配器','存储','U盘',
                '空调','冰箱','洗衣机','电视','微波炉','烤箱','吸尘器','净化','加湿',
                '智能','蓝牙','无线','高清','4K','游戏','办公','家用','便携','快充',
                '内存','硬盘','处理器','屏幕','像素','降噪'],
    '食品生鲜': ['零食','饼干','糕点','巧克力','糖果','坚果','果干','肉脯','海味','辣条',
                '水果','蔬菜','肉','鱼','虾','蟹','蛋','奶','豆','米','面','油',
                '茶','咖啡','果汁','饮料','酒','醋','酱','调料','盐','糖',
                '有机','绿色','新鲜','纯天然','无添加','低脂','低糖','即食','速食',
                '好吃','美味','香','脆','甜','辣','酸','鲜','营养','健康'],
    '美妆个护': ['口红','粉底','眼影','睫毛','眉笔','腮红','遮瑕','卸妆','洁面','面膜',
                '精华','乳液','面霜','防晒','隔离','水乳','爽肤','化妆','彩妆','护肤',
                '洗发','护发','沐浴','牙膏','牙刷','香水','指甲','美甲',
                '补水','保湿','美白','抗皱','控油','祛痘','修复','提亮','紧致',
                '天然','温和','敏感','清爽','滋润'],
    '家居日用': ['收纳','整理','置物','挂钩','衣架','垃圾桶','纸巾','毛巾','拖把','扫帚',
                '碗','盘','杯','筷','刀','锅','壶','瓶','罐','盆',
                '床','枕','被','毯','窗帘','地毯','沙发','桌','椅','灯','镜',
                '家用','日用','厨房','卫浴','客厅','卧室','阳台',
                '简约','创意','实用','多功能','便携','环保','防滑','防水','耐用'],
    '其他': [],
}
_ALL_PRODUCT_KEYWORDS = set()
for _kws in _CATEGORY_KEYWORDS.values():
    _ALL_PRODUCT_KEYWORDS.update(_kws)


def _score_title_quality(title_text, category):
    """Evaluate title on 3 dimensions: character quality, category relevance, info density."""
    if not title_text or len(title_text.strip()) < 2:
        return 0.1, {'char_quality': 10, 'relevance': 0, 'info_density': 10}

    chars = list(title_text)
    total = len(chars)
    unique = len(set(chars))
    unique_ratio = unique / total

    # Character Quality (anti-spam)
    max_consec = 1
    cur = 1
    for i in range(1, len(chars)):
        if chars[i] == chars[i - 1]:
            cur += 1
            if cur > max_consec:
                max_consec = cur
        else:
            cur = 1
    dominant_ratio = Counter(chars).most_common(1)[0][1] / total

    char_q = 1.0
    if unique_ratio < 0.3:
        char_q *= 0.15
    elif unique_ratio < 0.5:
        char_q *= 0.4
    elif unique_ratio < 0.7:
        char_q *= 0.75
    if max_consec >= 4:
        char_q *= 0.2
    elif max_consec >= 3:
        char_q *= 0.5
    if dominant_ratio > 0.5:
        char_q *= 0.2
    elif dominant_ratio > 0.35:
        char_q *= 0.5

    # Category Relevance
    cat_kws = _CATEGORY_KEYWORDS.get(category, [])
    if cat_kws:
        cat_matched = sum(1 for kw in cat_kws if kw in title_text)
        relevance = min(1.0, cat_matched / 3)
    else:
        general_matched = sum(1 for kw in _ALL_PRODUCT_KEYWORDS if kw in title_text)
        relevance = min(1.0, general_matched / 3)

    # Information Density
    content = re.sub(r'[\s\W]', '', title_text)
    effective = []
    for i, c in enumerate(content):
        if i == 0 or c != content[i - 1]:
            effective.append(c)
    density = len(effective) / total if total > 0 else 0

    combined = char_q * 0.35 + relevance * 0.40 + density * 0.25
    breakdown = {
        'char_quality': char_q * 100,
        'relevance': relevance * 100,
        'info_density': density * 100,
    }
    return combined, breakdown


def _score_title_length(length, cat_bench_length):
    optimal = cat_bench_length if cat_bench_length >= 15 else 25
    diff = abs(length - optimal)
    if diff <= 5:
        return 1.0
    elif diff <= 10:
        return 0.85
    elif diff <= 15:
        return 0.65
    elif diff <= 25:
        return 0.4
    else:
        return 0.2


def _score_sentiment(cal_emo):
    if cal_emo >= 0.7:
        return 1.0
    elif cal_emo >= 0.6:
        return 0.8 + (cal_emo - 0.6) * 2.0
    elif cal_emo >= 0.5:
        return 0.6 + (cal_emo - 0.5) * 2.0
    elif cal_emo >= 0.4:
        return 0.4 + (cal_emo - 0.4) * 2.0
    else:
        return max(0.1, cal_emo)


def _score_images(img_count):
    if 3 <= img_count <= 6:
        return 1.0
    elif img_count == 2 or img_count == 7:
        return 0.8
    elif img_count == 1 or img_count == 8:
        return 0.6
    elif img_count == 0:
        return 0.2
    else:
        return 0.5


def _score_video(has_video):
    return 1.0 if has_video else 0.4


def _score_price(price, cat_bench_price):
    if cat_bench_price <= 0:
        return 0.5
    ratio = price / cat_bench_price
    if ratio <= 0.8:
        return 1.0
    elif ratio <= 1.0:
        return 0.9
    elif ratio <= 1.2:
        return 0.7
    elif ratio <= 1.5:
        return 0.5
    else:
        return 0.3


def _score_discount(discount_rate):
    if 0.05 <= discount_rate <= 0.20:
        return 1.0
    elif 0.01 <= discount_rate < 0.05:
        return 0.7
    elif 0.20 < discount_rate <= 0.30:
        return 0.8
    elif discount_rate == 0:
        return 0.4
    else:
        return 0.5


def compute_domain_score(product_params, cat_bench, title_text=''):
    """Compute domain knowledge score (0-100) based on e-commerce best practices."""
    category = product_params.get('category', '')
    tq_combined, tq_breakdown = _score_title_quality(title_text, category)
    tl_score = _score_title_length(product_params['title_length'], cat_bench.get('title_length', 27))

    weights = {
        'title_quality': 0.25, 'title_length': 0.10, 'sentiment': 0.10,
        'images': 0.15, 'video': 0.15, 'price': 0.15, 'discount': 0.10,
    }
    scores = {
        'title_quality': tq_combined,
        'title_length': tl_score,
        'sentiment': _score_sentiment(product_params['title_emo_score']),
        'images': _score_images(product_params['img_count']),
        'video': _score_video(product_params['has_video']),
        'price': _score_price(product_params['price'], cat_bench.get('price', 80)),
        'discount': _score_discount(product_params['discount_rate']),
    }
    total = sum(scores[k] * weights[k] for k in weights)
    breakdown = {k: scores[k] * 100 for k in scores}
    breakdown['_tq_detail'] = tq_breakdown
    return total * 100, breakdown


def compute_hybrid_score(ml_prob, domain_score, ml_weight=0.5, domain_weight=0.5):
    baseline = df['label'].mean()
    ml_score = min(100, (ml_prob / baseline) * 50)
    hybrid = ml_weight * ml_score + domain_weight * domain_score
    return hybrid, ml_score


# ─── Sidebar ───
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Product Analyzer", "User Clusters"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model**: {artifacts['model_name']}")
st.sidebar.markdown(f"**AUC**: {comparison_df['auc'].max():.4f}")
st.sidebar.markdown(f"**Samples**: {len(df):,}")
st.sidebar.markdown(f"**Categories**: {len(category_mapping)}")
st.sidebar.markdown(f"**Clusters**: {OPTIMAL_K}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Sentiment**: SnowNLP + Beta QM")
st.sidebar.caption("SnowNLP raw scores are calibrated via Beta quantile mapping to match training data distribution")


# ═══════════════════════════════════════════════
# PAGE 1: Product Analyzer (Consolidated)
# ═══════════════════════════════════════════════
if page == "Product Analyzer":
    st.markdown("# Product Analyzer")
    st.markdown("Input product features to get purchase prediction, quality analysis, user persona insights, and actionable seller recommendations")

    # ── Input Section ──
    st.markdown('<div class="section-header">Product Information</div>', unsafe_allow_html=True)

    product_title = st.text_input(
        "Product Title (Chinese)",
        value="",
        placeholder="e.g., 夏季新款时尚女装连衣裙 轻薄透气修身显瘦百搭"
    )

    # Auto-compute title features (hidden from user)
    if product_title.strip():
        auto_title_length = len(product_title.strip())
        raw_emo, cal_emo = compute_sentiment(product_title.strip())
    else:
        auto_title_length = 25
        raw_emo, cal_emo = 0.5, 0.5

    # Product features
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", list(category_mapping.keys()))
        img_count = st.slider("Image Count", 0, 10, 3)
    with col2:
        price = st.number_input("Price", 1.0, 1000.0, 80.0, step=5.0)
        discount_rate = st.slider("Discount Rate", 0.0, 0.5, 0.1, step=0.01)
    with col3:
        has_video = st.selectbox("Has Video?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        coupon = st.selectbox("Offer Coupon?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    analyze_btn = st.button("Analyze Product", type="primary", use_container_width=True)

    if analyze_btn:
        st.markdown("---")

        product_params = {
            'title_length': auto_title_length,
            'title_emo_score': cal_emo,
            'img_count': img_count,
            'has_video': has_video,
            'price': price,
            'discount_rate': discount_rate,
            'category': category,
        }
        cat_bench = benchmarks[category]

        # Cluster-weighted prediction
        prob, per_cluster = predict_cluster_weighted(product_params, feature_cols, category_mapping, coupon)

        # Build a feature vector for counterfactual analysis (using global median as base)
        user_params = get_default_user_params()
        user_params['coupon_received'] = coupon
        feature_vec = build_feature_vector(product_params, user_params, feature_cols, category_mapping)

        # ════════════════════════════════════════════
        # Section 1: Prediction Results
        # ════════════════════════════════════════════
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

        col_pred, col_hybrid = st.columns(2)

        with col_pred:
            # Gauge: cluster-weighted purchase probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 28}},
                title={'text': "Cluster-Weighted Purchase Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#667eea'},
                    'steps': [
                        {'range': [0, 30], 'color': '#ffebee'},
                        {'range': [30, 60], 'color': '#fff8e1'},
                        {'range': [60, 100], 'color': '#e8f5e9'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 2},
                        'thickness': 0.75, 'value': 50
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_hybrid:
            domain_total, domain_breakdown = compute_domain_score(product_params, cat_bench, product_title)
            hybrid, ml_score = compute_hybrid_score(prob, domain_total)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hybrid,
                number={'font': {'size': 28}},
                title={'text': "Hybrid Score (ML + Domain Knowledge)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#ff9800'},
                    'steps': [
                        {'range': [0, 40], 'color': '#ffebee'},
                        {'range': [40, 70], 'color': '#fff8e1'},
                        {'range': [70, 100], 'color': '#e8f5e9'}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Per-cluster prediction breakdown
        st.markdown("**Per-Cluster Purchase Probability:**")
        c_labels = [f"C{c}: {per_cluster[c]['name']}" for c in per_cluster]
        c_probs = [per_cluster[c]['prob'] * 100 for c in per_cluster]
        c_weights_pct = [per_cluster[c]['weight'] * 100 for c in per_cluster]
        c_colors = ['#667eea', '#f5576c', '#43e97b', '#f39c12']

        fig_cl = go.Figure()
        fig_cl.add_trace(go.Bar(
            x=[f"C{c}" for c in per_cluster], y=c_probs,
            marker_color=c_colors[:len(c_probs)],
            text=[f'{p:.1f}%' for p in c_probs],
            textposition='outside',
            name='Purchase Prob'
        ))
        fig_cl.update_layout(
            yaxis_title="Purchase Probability (%)",
            height=220, margin=dict(t=10, b=10, l=40, r=10),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cl, use_container_width=True)

        # Cluster legend
        cluster_info_cols = st.columns(OPTIMAL_K)
        for i, ci_col in enumerate(cluster_info_cols):
            with ci_col:
                st.caption(f"C{i}: {cluster_names[i]} ({cluster_weights.get(i, 0)*100:.1f}% of users)")

        # ════════════════════════════════════════════
        # Section 2: Domain Score Breakdown + Title Quality
        # ════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="section-header">Domain Score Breakdown</div>', unsafe_allow_html=True)

        col_domain, col_tq = st.columns(2)

        with col_domain:
            main_keys = [k for k in domain_breakdown if not k.startswith('_')]
            bd_label_map = {
                'title_quality': 'Title Quality (25%)',
                'title_length': 'Title Length (10%)',
                'sentiment': 'Sentiment (10%)',
                'images': 'Images (15%)',
                'video': 'Video (15%)',
                'price': 'Price (15%)',
                'discount': 'Discount (10%)',
            }
            bd_display = [bd_label_map.get(k, k) for k in main_keys]
            bd_vals = [domain_breakdown[k] for k in main_keys]
            bd_colors = ['#4caf50' if v >= 70 else ('#ff9800' if v >= 40 else '#f44336') for v in bd_vals]

            fig2 = go.Figure(go.Bar(
                x=bd_vals, y=bd_display, orientation='h',
                marker_color=bd_colors,
                text=[f'{v:.0f}' for v in bd_vals],
                textposition='outside'
            ))
            fig2.update_layout(
                title=f"Domain Score: {domain_total:.0f}/100",
                xaxis_range=[0, 115], height=300,
                margin=dict(l=140, t=35, b=10, r=10),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_tq:
            tq_detail = domain_breakdown.get('_tq_detail', {})
            if tq_detail:
                tq_labels = ['Char Quality (anti-spam)', 'Category Relevance', 'Info Density']
                tq_keys = ['char_quality', 'relevance', 'info_density']
                tq_vals = [tq_detail.get(k, 0) for k in tq_keys]
                tq_colors = ['#4caf50' if v >= 70 else ('#ff9800' if v >= 40 else '#f44336') for v in tq_vals]

                fig3 = go.Figure(go.Bar(
                    x=tq_vals, y=tq_labels, orientation='h',
                    marker_color=tq_colors,
                    text=[f'{v:.0f}' for v in tq_vals],
                    textposition='outside'
                ))
                fig3.update_layout(
                    title="Title Quality Detail",
                    xaxis_range=[0, 115], height=200,
                    margin=dict(l=170, t=35, b=10, r=10),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Popularity estimate
            if category in pop_stats:
                ps = pop_stats[category]
                cat_avg_rate = df[df['category'] == category]['label'].mean()
                ratio = prob / cat_avg_rate if cat_avg_rate > 0 else 1.0
                like_est = ps['like_num_median'] * ratio
                collect_est = ps['collect_num_median'] * ratio

                st.markdown(f"""<div class="pop-card">
                    <h4>Estimated Engagement ({category})</h4>
                    <p>Estimated likes: <b>~{like_est:.0f}</b> | collections: <b>~{collect_est:.0f}</b></p>
                    <p>Category median: likes {ps['like_num_median']:.0f} / collections {ps['collect_num_median']:.0f}</p>
                </div>""", unsafe_allow_html=True)

        # ════════════════════════════════════════════
        # Section 3: Counterfactual Analysis
        # ════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="section-header">Counterfactual Analysis</div>', unsafe_allow_html=True)
        st.caption("Shows expected purchase rate change if each feature is changed to category benchmark value")

        cf_results = []
        for feat in SELLER_CONTROLLABLE:
            current_val = product_params[feat]
            ideal_val = cat_bench[feat]
            _, new_prob, delta = simulate_counterfactual(
                model, pd.Series(feature_vec, index=feature_cols),
                feat, ideal_val, feature_cols
            )
            cf_results.append({
                'feature': feat,
                'current': current_val,
                'benchmark': ideal_val,
                'delta': delta
            })

        cf_df = pd.DataFrame(cf_results).sort_values('delta', ascending=False)

        fig = go.Figure()
        colors_bar = ['#00c853' if d > 0 else '#ff5252' for d in cf_df['delta']]
        fig.add_trace(go.Bar(
            y=cf_df['feature'], x=cf_df['delta'] * 100,
            orientation='h', marker_color=colors_bar,
            text=[f'{d*100:+.2f}%' for d in cf_df['delta']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Expected Purchase Rate Change (current -> category benchmark)",
            xaxis_title="Probability Change (%)",
            height=250, margin=dict(l=130, t=40),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ════════════════════════════════════════════
        # Section 4: Target User Persona + Recommendations
        # ════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="section-header">Target User Persona & Seller Recommendations</div>', unsafe_allow_html=True)

        col_persona, col_recs = st.columns([1, 1.5])

        with col_persona:
            if 'user_cluster' in df.columns:
                cat_cluster_rates = df[df['category'] == category].groupby('user_cluster')['label'].mean()
                if len(cat_cluster_rates) > 0:
                    best_cluster = cat_cluster_rates.idxmax()
                    best_rate = cat_cluster_rates.max()
                    bp = cluster_profiles.loc[best_cluster]

                    st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#ff6f00 0%,#ff8f00 100%);text-align:left;padding:1.2rem">
                        <h3>Best Segment for {category}</h3>
                        <h1 style="font-size:1.2rem">{cluster_names[best_cluster]}</h1>
                        <p style="margin-top:0.8rem;font-size:0.85rem;opacity:0.95">
                        Purchase rate: <b>{best_rate:.1%}</b><br>
                        Avg age: <b>{bp['age']:.0f}</b> |
                        Gender: <b>{'Male' if bp['gender'] > 0.5 else 'Female'}</b><br>
                        Avg spend: <b>{bp['total_spend']:,.0f}</b> |
                        Level: <b>{bp['user_level']:.1f}</b><br>
                        Freq: <b>{bp['purchase_freq']:.0f}</b> |
                        <b>{int(bp['count']):,}</b> users
                        </p>
                    </div>""", unsafe_allow_html=True)

                    # Category purchase rate by cluster bar chart
                    fig_cp = go.Figure()
                    colors_cl = ['#667eea', '#f5576c', '#43e97b', '#f39c12']
                    for ci in range(OPTIMAL_K):
                        rate = cat_cluster_rates.get(ci, 0)
                        fig_cp.add_trace(go.Bar(
                            x=[f'C{ci}'], y=[rate * 100],
                            name=f'C{ci}',
                            marker_color=colors_cl[ci],
                            text=[f'{rate*100:.1f}%'],
                            textposition='outside'
                        ))
                    fig_cp.update_layout(
                        title=f"Purchase Rate by Cluster ({category})",
                        yaxis_title="Purchase Rate (%)",
                        height=250, margin=dict(t=40, b=20),
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_cp, use_container_width=True)

        with col_recs:
            st.markdown("**Actionable Recommendations:**")

            cards = []
            title_length = product_params['title_length']
            title_emo_score = product_params['title_emo_score']

            # Title quality recommendation
            tq_score = domain_breakdown.get('title_quality', 0)
            if tq_score < 60:
                cards.append(f"""<div class="rec-card rec-card-negative">
                    <h4>Title Quality: Needs Improvement</h4>
                    <p>Title quality score: {tq_score:.0f}/100. Consider using more relevant keywords, avoiding repetitive characters, and increasing information density.</p>
                </div>""")

            # Title length
            if title_length < cat_bench['title_length'] * 0.8:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'title_length', cat_bench['title_length'], feature_cols
                )
                cards.append(f"""<div class="rec-card rec-card-positive">
                    <h4>Title: Increase Length</h4>
                    <p>Current: {title_length} chars | Benchmark: {cat_bench['title_length']:.0f} chars</p>
                    <p>Add more descriptive keywords for better discoverability. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")
            elif title_length > cat_bench['title_length'] * 1.5:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'title_length', cat_bench['title_length'], feature_cols
                )
                cards.append(f"""<div class="rec-card rec-card-negative">
                    <h4>Title: Too Long</h4>
                    <p>Current: {title_length} chars | Benchmark: {cat_bench['title_length']:.0f} chars</p>
                    <p>Shorten for readability. Focus on key selling points. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")

            # Sentiment
            if title_emo_score < cat_bench['title_emo_score'] * 0.7:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'title_emo_score', cat_bench['title_emo_score'], feature_cols
                )
                cards.append(f"""<div class="rec-card rec-card-positive">
                    <h4>Title: Boost Emotional Appeal</h4>
                    <p>Current: {title_emo_score:.2f} | Benchmark: {cat_bench['title_emo_score']:.2f}</p>
                    <p>Use more engaging, positive language. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")

            # Video
            if has_video == 0:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'has_video', 1, feature_cols
                )
                cards.append(f"""<div class="rec-card">
                    <h4>Content: Add Product Video</h4>
                    <p>Video content significantly boosts conversion. Consider demo or unboxing. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")

            # Images
            if img_count < cat_bench['img_count'] * 0.7:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'img_count', cat_bench['img_count'], feature_cols
                )
                cards.append(f"""<div class="rec-card rec-card-positive">
                    <h4>Content: More Images</h4>
                    <p>Current: {img_count} | Benchmark: {cat_bench['img_count']:.0f}. Add multi-angle product shots. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")

            # Price
            if price > cat_bench['price'] * 1.3:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'price', cat_bench['price'], feature_cols
                )
                cards.append(f"""<div class="rec-card rec-card-negative">
                    <h4>Price: Above Category Median</h4>
                    <p>Current: {price:.1f} | Category median: {cat_bench['price']:.1f}</p>
                    <p>Consider price reduction or bundle offers. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")

            # Discount
            if discount_rate < cat_bench['discount_rate'] * 0.5 and cat_bench['discount_rate'] > 0:
                _, _, delta = simulate_counterfactual(
                    model, pd.Series(feature_vec, index=feature_cols),
                    'discount_rate', cat_bench['discount_rate'], feature_cols
                )
                cards.append(f"""<div class="rec-card rec-card-positive">
                    <h4>Discount: Consider Offering Discount</h4>
                    <p>Current: {discount_rate:.0%} | Category avg: {cat_bench['discount_rate']:.0%}</p>
                    <p>A moderate 5-20% discount can boost conversion. <b>Expected lift: {delta*100:+.2f}%</b></p>
                </div>""")

            # Target audience
            if 'user_cluster' in df.columns and len(cat_cluster_rates) > 0:
                cards.append(f"""<div class="rec-card">
                    <h4>Target Audience</h4>
                    <p>Focus marketing on <b>{cluster_names[best_cluster]}</b> segment ({best_rate:.1%} purchase rate for {category}).</p>
                    <p>This segment has {int(cluster_profiles.loc[best_cluster, 'count']):,} users.</p>
                </div>""")

            if cards:
                for card in cards:
                    st.markdown(card, unsafe_allow_html=True)
            else:
                st.success("Product features are well-aligned with category benchmarks. No major improvements needed.")

        # ════════════════════════════════════════════
        # Section 5: Category Benchmarks Reference
        # ════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="section-header">Category Benchmark Reference</div>', unsafe_allow_html=True)

        bench_data = {
            'Feature': SELLER_CONTROLLABLE,
            'Your Value': [product_params[f] for f in SELLER_CONTROLLABLE],
            'Category Benchmark': [cat_bench[f] for f in SELLER_CONTROLLABLE],
        }
        bench_df = pd.DataFrame(bench_data)
        bench_df['Status'] = bench_df.apply(
            lambda r: 'Above' if r['Your Value'] > r['Category Benchmark'] * 1.1
            else ('Below' if r['Your Value'] < r['Category Benchmark'] * 0.9 else 'On target'),
            axis=1
        )
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    else:
        st.info("Enter a product title and configure features above, then click **Analyze Product** to get comprehensive analysis.")


# ═══════════════════════════════════════════════
# PAGE 2: User Clusters
# ═══════════════════════════════════════════════
elif page == "User Clusters":
    st.markdown("# User Cluster Analysis")
    st.markdown("Understanding user segments for targeted marketing")

    if 'user_cluster' not in df.columns:
        st.warning("Run the notebook first to generate cluster assignments.")
        st.stop()

    # ── Cluster Overview Cards ──
    cols = st.columns(OPTIMAL_K)
    colors_cluster = ['#667eea', '#f5576c', '#43e97b', '#f39c12']
    for i, col in enumerate(cols):
        with col:
            pr = cluster_profiles.loc[i, 'purchase_rate']
            cnt = int(cluster_profiles.loc[i, 'count'])
            pct = cnt / len(df) * 100
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,{colors_cluster[i]} 0%,{colors_cluster[i]}88 100%)">
                <h3>Cluster {i}</h3>
                <h1>{pr:.1%}</h1>
                <p style="margin:0;font-size:0.75rem;opacity:0.8">{cluster_names[i]}<br>{cnt:,} users ({pct:.1f}%)</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Cluster Feature Profiles</div>', unsafe_allow_html=True)

        radar_features = ['age', 'user_level', 'purchase_freq', 'total_spend', 'purchase_intent', 'add2cart']
        available_radar = [f for f in radar_features if f in df.columns]
        cluster_means = df.groupby('user_cluster')[available_radar].mean()
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

        fig = go.Figure()
        for i in range(OPTIMAL_K):
            vals = cluster_means_norm.loc[i].tolist()
            vals.append(vals[0])
            cats = available_radar + [available_radar[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats,
                fill='toself', name=f'C{i}: {cluster_names[i]}',
                line_color=colors_cluster[i], opacity=0.7
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450, title="Normalized Feature Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Marketing Effectiveness</div>', unsafe_allow_html=True)

        marketing_data = []
        for c in range(OPTIMAL_K):
            c_mask = df['user_cluster'] == c
            coupon_mask = df['coupon_received'] == 1
            no_coupon_mask = df['coupon_received'] == 0

            pr_coupon = df[c_mask & coupon_mask]['label'].mean() if (c_mask & coupon_mask).sum() > 0 else 0
            pr_no_coupon = df[c_mask & no_coupon_mask]['label'].mean() if (c_mask & no_coupon_mask).sum() > 0 else 0
            pr_video = df[c_mask & (df['has_video'] == 1)]['label'].mean() if (c_mask & (df['has_video'] == 1)).sum() > 0 else 0
            pr_no_video = df[c_mask & (df['has_video'] == 0)]['label'].mean() if (c_mask & (df['has_video'] == 0)).sum() > 0 else 0

            marketing_data.append({
                'Cluster': f'C{c}',
                'Coupon Lift': pr_coupon - pr_no_coupon,
                'Video Lift': pr_video - pr_no_video,
            })

        mkt_df = pd.DataFrame(marketing_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Coupon Lift', x=mkt_df['Cluster'],
            y=mkt_df['Coupon Lift'] * 100,
            marker_color='#667eea',
            text=[f'{v*100:+.1f}%' for v in mkt_df['Coupon Lift']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='Video Lift', x=mkt_df['Cluster'],
            y=mkt_df['Video Lift'] * 100,
            marker_color='#f5576c',
            text=[f'{v*100:+.1f}%' for v in mkt_df['Video Lift']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Purchase Rate Lift by Marketing Action",
            yaxis_title="Purchase Rate Lift (%)",
            barmode='group', height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Detailed Profiles Table ──
    st.markdown('<div class="section-header">Detailed Cluster Profiles</div>', unsafe_allow_html=True)

    display_cols = [c for c in ['age', 'gender', 'user_level', 'purchase_freq', 'total_spend',
                                'purchase_intent', 'add2cart', 'purchase_rate', 'count']
                    if c in cluster_profiles.columns]
    display_df = cluster_profiles[display_cols].copy()
    display_df.index = [f'C{i}: {cluster_names[i]}' for i in display_df.index]
    if 'purchase_rate' in display_df.columns:
        display_df['purchase_rate'] = display_df['purchase_rate'].apply(lambda x: f'{x:.1%}')
    if 'count' in display_df.columns:
        display_df['count'] = display_df['count'].apply(lambda x: f'{x:,.0f}')
    if 'total_spend' in display_df.columns:
        display_df['total_spend'] = display_df['total_spend'].apply(lambda x: f'{x:,.0f}')
    st.dataframe(display_df, use_container_width=True)

    # ── Purchase Rate by Category x Cluster ──
    st.markdown('<div class="section-header">Purchase Rate by Category x Cluster</div>', unsafe_allow_html=True)

    cat_cluster = df.groupby(['category', 'user_cluster'])['label'].mean().reset_index()
    cat_cluster.columns = ['category', 'cluster', 'purchase_rate']
    cat_cluster['cluster_name'] = cat_cluster['cluster'].map(lambda x: f'C{x}: {cluster_names.get(x, "")}')

    fig = px.bar(
        cat_cluster, x='category', y='purchase_rate', color='cluster_name',
        barmode='group', color_discrete_sequence=colors_cluster,
        text=cat_cluster['purchase_rate'].apply(lambda x: f'{x:.1%}')
    )
    fig.update_layout(
        title="Purchase Rate by Category and Cluster",
        yaxis_title="Purchase Rate",
        height=400, legend_title="Cluster"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
