import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from snownlp import SnowNLP
from scipy import stats

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
    .ab-better { background: #e8f5e9; border: 2px solid #4caf50; }
    .ab-worse { background: #ffebee; border: 2px solid #ef5350; }
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
    """Predict with proper feature names to avoid sklearn warnings."""
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


# ─── Domain Knowledge Scoring ───
# E-commerce best practices: title quality, content richness, pricing strategy
# Each sub-score is 0-1, combined via weighted average.

# Category keyword dictionaries for relevance scoring
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
    """Evaluate title on 3 dimensions: character quality, category relevance, info density.
    Returns (combined 0-1, breakdown dict with 3 sub-scores 0-100)."""
    import re
    from collections import Counter

    if not title_text or len(title_text.strip()) < 2:
        return 0.1, {'char_quality': 10, 'relevance': 0, 'info_density': 10}

    chars = list(title_text)
    total = len(chars)
    unique = len(set(chars))
    unique_ratio = unique / total

    # --- Character Quality (anti-spam) ---
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

    # --- Category Relevance ---
    cat_kws = _CATEGORY_KEYWORDS.get(category, [])
    if cat_kws:
        cat_matched = sum(1 for kw in cat_kws if kw in title_text)
        relevance = min(1.0, cat_matched / 3)
    else:
        general_matched = sum(1 for kw in _ALL_PRODUCT_KEYWORDS if kw in title_text)
        relevance = min(1.0, general_matched / 3)

    # --- Information Density ---
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
    """Optimal title: 15-35 chars. Too short hurts SEO, too long hurts readability."""
    optimal = cat_bench_length
    if optimal < 15:
        optimal = 25
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
    """Higher positive sentiment is better for e-commerce titles."""
    # Sigmoid-like: ramp from 0.3 to 0.8 maps to score 0.3 to 1.0
    if cal_emo >= 0.7:
        return 1.0
    elif cal_emo >= 0.6:
        return 0.8 + (cal_emo - 0.6) * 2.0  # 0.8 -> 1.0
    elif cal_emo >= 0.5:
        return 0.6 + (cal_emo - 0.5) * 2.0  # 0.6 -> 0.8
    elif cal_emo >= 0.4:
        return 0.4 + (cal_emo - 0.4) * 2.0  # 0.4 -> 0.6
    else:
        return max(0.1, cal_emo)


def _score_images(img_count):
    """3-6 images is optimal for most e-commerce platforms."""
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
    """Having a video boosts conversion significantly in e-commerce."""
    return 1.0 if has_video else 0.4


def _score_price(price, cat_bench_price):
    """Price at or below category benchmark is competitive."""
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
    """Moderate discount (5-20%) is optimal; no discount or extreme discount is suboptimal."""
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
    """Compute domain knowledge score (0-100) based on e-commerce best practices.
    Returns (total_score, breakdown_dict).
    title_text: raw title string for quality/relevance analysis."""
    category = product_params.get('category', '')

    # Title quality: char quality + relevance + info density (replaces simple length + sentiment)
    tq_combined, tq_breakdown = _score_title_quality(title_text, category)
    tl_score = _score_title_length(product_params['title_length'], cat_bench.get('title_length', 27))

    weights = {
        'title_quality': 0.25,
        'title_length': 0.10,
        'sentiment': 0.10,
        'images': 0.15,
        'video': 0.15,
        'price': 0.15,
        'discount': 0.10,
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
    # Attach title quality sub-breakdown for detailed display
    breakdown['_tq_detail'] = tq_breakdown
    return total * 100, breakdown


def compute_hybrid_score(ml_prob, domain_score, ml_weight=0.5, domain_weight=0.5):
    """Combine ML prediction and domain knowledge into a hybrid score (0-100).
    ML prob is mapped to 0-100 scale relative to baseline purchase rate."""
    # Normalize ML prob: dataset avg is ~45%, map to a 0-100 quality scale
    # 0% prob -> 0, 45% -> 50, 90% -> 100
    baseline = df['label'].mean()
    ml_score = min(100, (ml_prob / baseline) * 50)
    hybrid = ml_weight * ml_score + domain_weight * domain_score
    return hybrid, ml_score


def generate_recommendations_html(product_params, cat_bench, feature_vec, category):
    """Generate recommendation cards as HTML."""
    cards = []
    title_length = product_params['title_length']
    title_emo_score = product_params['title_emo_score']
    img_count = product_params['img_count']
    has_video = product_params['has_video']
    price = product_params['price']
    discount_rate = product_params['discount_rate']

    # Title length
    if title_length < cat_bench['title_length'] * 0.8:
        _, _, delta = simulate_counterfactual(
            model, pd.Series(feature_vec, index=feature_cols),
            'title_length', cat_bench['title_length'], feature_cols
        )
        cards.append(f"""<div class="rec-card rec-card-positive">
            <h4>Title: Increase Length</h4>
            <p>Current: {title_length} chars | Benchmark: {cat_bench['title_length']:.0f} chars</p>
            <p>Add more descriptive keywords to improve discoverability.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
        </div>""")
    elif title_length > cat_bench['title_length'] * 1.5:
        _, _, delta = simulate_counterfactual(
            model, pd.Series(feature_vec, index=feature_cols),
            'title_length', cat_bench['title_length'], feature_cols
        )
        cards.append(f"""<div class="rec-card rec-card-negative">
            <h4>Title: Too Long</h4>
            <p>Current: {title_length} chars | Benchmark: {cat_bench['title_length']:.0f} chars</p>
            <p>Shorten for better readability. Focus on key selling points.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
        </div>""")

    # Emotion score
    if title_emo_score < cat_bench['title_emo_score'] * 0.7:
        _, _, delta = simulate_counterfactual(
            model, pd.Series(feature_vec, index=feature_cols),
            'title_emo_score', cat_bench['title_emo_score'], feature_cols
        )
        cards.append(f"""<div class="rec-card rec-card-positive">
            <h4>Title: Boost Emotion</h4>
            <p>Current: {title_emo_score:.2f} | Benchmark: {cat_bench['title_emo_score']:.2f}</p>
            <p>Use more engaging, emotional language in your title.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
        </div>""")

    # Video
    if has_video == 0:
        _, _, delta = simulate_counterfactual(
            model, pd.Series(feature_vec, index=feature_cols),
            'has_video', 1, feature_cols
        )
        cards.append(f"""<div class="rec-card">
            <h4>Content: Add Video</h4>
            <p>Consider adding a product demonstration or unboxing video.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
        </div>""")

    # Image count
    if img_count < cat_bench['img_count'] * 0.7:
        _, _, delta = simulate_counterfactual(
            model, pd.Series(feature_vec, index=feature_cols),
            'img_count', cat_bench['img_count'], feature_cols
        )
        cards.append(f"""<div class="rec-card rec-card-positive">
            <h4>Content: More Images</h4>
            <p>Current: {img_count} | Benchmark: {cat_bench['img_count']:.0f}</p>
            <p>Add more high-quality product images from different angles.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
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
            <p>Consider price reduction, bundle offers, or value-added services.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
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
            <p>A moderate discount can significantly boost conversion.</p>
            <p><b>Expected lift: {delta*100:+.2f}%</b></p>
        </div>""")

    # Target audience
    if 'user_cluster' in df.columns:
        cat_cluster_rates = df[df['category'] == category].groupby('user_cluster')['label'].mean()
        best_cluster = cat_cluster_rates.idxmax()
        best_rate = cat_cluster_rates.max()
        cards.append(f"""<div class="rec-card">
            <h4>Target Audience</h4>
            <p>Best user segment: Cluster {best_cluster} ({cluster_names[best_cluster]})</p>
            <p>Purchase rate for {category}: {best_rate:.1%}. Focus marketing on this segment.</p>
        </div>""")

    return cards


# ─── Sidebar ───
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Product Analyzer", "Title A/B Test", "User Clusters", "Category Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model**: {artifacts['model_name']}")
st.sidebar.markdown(f"**AUC**: {comparison_df['auc'].max():.4f}")
st.sidebar.markdown(f"**Samples**: {len(df):,}")
st.sidebar.markdown(f"**Categories**: {len(category_mapping)}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Sentiment**: SnowNLP")
st.sidebar.markdown("**Calibration**: Beta QM")
st.sidebar.caption("SnowNLP raw scores are calibrated via Beta quantile mapping to match the training data distribution")


# ═══════════════════════════════════════════════
# PAGE 1: Dashboard
# ═══════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("# Seller Recommendation System")
    st.markdown("Data-driven recommendations to improve product purchase rates")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>Total Records</h3><h1>{len(df):,}</h1>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%)">
            <h3>Purchase Rate</h3><h1>{df['label'].mean():.1%}</h1>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%)">
            <h3>Best AUC</h3><h1>{comparison_df['auc'].max():.4f}</h1>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%)">
            <h3>Categories</h3><h1>{len(category_mapping)}</h1>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        colors_m = ['#667eea', '#f5576c', '#43e97b']
        for i, (name, row) in enumerate(comparison_df.iterrows()):
            fig.add_trace(go.Bar(
                name=name, x=metrics_list, y=[row[m] for m in metrics_list],
                marker_color=colors_m[i], opacity=0.9
            ))
        fig.update_layout(
            title="Metrics Comparison", barmode='group',
            yaxis_range=[0, 1], height=400,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Table(
            header=dict(
                values=['Model'] + [m.upper() for m in metrics_list],
                fill_color='#667eea', font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=[
                    comparison_df.index.tolist(),
                    *[comparison_df[m].apply(lambda x: f'{x:.4f}').tolist() for m in metrics_list]
                ],
                fill_color=[['#f8f9fa', '#ffffff'] * 2],
                align='center', font=dict(size=11)
            )
        ))
        fig.update_layout(title="Detailed Metrics", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Purchase Rate by Category</div>', unsafe_allow_html=True)
    cat_stats = df.groupby('category').agg(
        purchase_rate=('label', 'mean'),
        count=('label', 'count'),
        avg_price=('price', 'mean')
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            cat_stats.sort_values('purchase_rate', ascending=False),
            x='category', y='purchase_rate',
            color='purchase_rate', color_continuous_scale='Viridis',
            text=cat_stats.sort_values('purchase_rate', ascending=False)['purchase_rate'].apply(lambda x: f'{x:.1%}')
        )
        fig.update_layout(title="Purchase Rate", height=350, showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            cat_stats.sort_values('avg_price', ascending=False),
            x='category', y='avg_price',
            color='avg_price', color_continuous_scale='Magma',
            text=cat_stats.sort_values('avg_price', ascending=False)['avg_price'].apply(lambda x: f'{x:.0f}')
        )
        fig.update_layout(title="Average Price", height=350, showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2: Product Analyzer (Enhanced)
# ═══════════════════════════════════════════════
elif page == "Product Analyzer":
    st.markdown("# Product Analyzer")
    st.markdown("Enter a product title and features to get purchase prediction, popularity estimate, user persona, and recommendations")

    # ── Input Section ──
    st.markdown('<div class="section-header">Product Information</div>', unsafe_allow_html=True)

    # Title input with auto sentiment
    product_title = st.text_input(
        "Product Title (Chinese)",
        value="",
        placeholder="e.g., 夏季新款时尚女装连衣裙 轻薄透气修身显瘦百搭"
    )

    # Auto-compute title features
    if product_title.strip():
        auto_title_length = len(product_title.strip())
        raw_emo, cal_emo = compute_sentiment(product_title.strip())
    else:
        auto_title_length = 25
        raw_emo, cal_emo = 0.5, 0.5

    # Show auto-computed values
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(f"""<div class="title-card">
            <h4>Auto-detected Title Length</h4>
            <span class="score">{auto_title_length}</span> characters
        </div>""", unsafe_allow_html=True)
    with col_info2:
        raw_color = '#4caf50' if raw_emo > 0.6 else ('#ff9800' if raw_emo > 0.4 else '#f44336')
        st.markdown(f"""<div class="title-card">
            <h4>SnowNLP Raw Score</h4>
            <span class="score" style="color:{raw_color}">{raw_emo:.3f}</span>
            <span style="color:#999;font-size:0.8rem"> (original)</span>
        </div>""", unsafe_allow_html=True)
    with col_info3:
        cal_color = '#4caf50' if cal_emo > 0.6 else ('#ff9800' if cal_emo > 0.4 else '#f44336')
        st.markdown(f"""<div class="title-card">
            <h4>Calibrated Score (used by model)</h4>
            <span class="score" style="color:{cal_color}">{cal_emo:.3f}</span>
            <span style="color:#999;font-size:0.8rem"> (Beta quantile mapped)</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

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
        user_params = get_default_user_params()
        user_params['coupon_received'] = coupon
        feature_vec = build_feature_vector(product_params, user_params, feature_cols, category_mapping)

        # ── Row 1: ML Prediction + Hybrid Score + Popularity ──
        col_pred, col_domain, col_pop = st.columns(3)

        cat_bench = benchmarks[category]

        with col_pred:
            st.markdown('<div class="section-header">ML Model Prediction</div>', unsafe_allow_html=True)
            prob = predict_proba_safe(model, feature_vec, feature_cols)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 30}},
                title={'text': "Purchase Probability"},
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
            fig.update_layout(height=260, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_domain:
            st.markdown('<div class="section-header">Hybrid Score</div>', unsafe_allow_html=True)

            domain_total, domain_breakdown = compute_domain_score(product_params, cat_bench, product_title)
            hybrid, ml_score = compute_hybrid_score(prob, domain_total)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hybrid,
                number={'suffix': '', 'font': {'size': 30}},
                title={'text': "ML + Domain Knowledge"},
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
            fig.update_layout(height=260, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Domain breakdown
            # Main dimensions (exclude _tq_detail)
            main_keys = [k for k in domain_breakdown if not k.startswith('_')]
            bd_label_map = {
                'title_quality': 'Title Quality',
                'title_length': 'Title Length',
                'sentiment': 'Sentiment',
                'images': 'Images',
                'video': 'Video',
                'price': 'Price',
                'discount': 'Discount',
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
                title="Domain Score Breakdown",
                xaxis_range=[0, 115], height=250,
                margin=dict(l=100, t=30, b=10, r=10),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Title quality sub-breakdown
            tq_detail = domain_breakdown.get('_tq_detail', {})
            if tq_detail:
                tq_labels = ['Char Quality', 'Category Relevance', 'Info Density']
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
                    xaxis_range=[0, 115], height=170,
                    margin=dict(l=120, t=30, b=10, r=10),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig3, use_container_width=True)

        with col_pop:
            st.markdown('<div class="section-header">Estimated Popularity</div>', unsafe_allow_html=True)

            if category in pop_stats:
                ps = pop_stats[category]
                # Estimate relative position based on product features vs category stats
                cat_data_raw = df_raw[df_raw['category'] == category]

                # Simple heuristic: products with higher predicted purchase prob tend to be more popular
                like_est = ps['like_num_median']
                collect_est = ps['collect_num_median']

                # Adjust by purchase probability relative to category average
                cat_avg_rate = df[df['category'] == category]['label'].mean()
                ratio = prob / cat_avg_rate if cat_avg_rate > 0 else 1.0

                like_est_adj = like_est * ratio
                collect_est_adj = collect_est * ratio

                # Show as percentile within category
                like_pct = (cat_data_raw['like_num'] <= like_est_adj).mean() * 100
                collect_pct = (cat_data_raw['collect_num'] <= collect_est_adj).mean() * 100

                st.markdown(f"""<div class="pop-card">
                    <h4>Estimated Engagement (vs {category})</h4>
                    <p>Estimated likes: <b>~{like_est_adj:.0f}</b> (top {100-like_pct:.0f}% in category)</p>
                    <p>Estimated collections: <b>~{collect_est_adj:.0f}</b> (top {100-collect_pct:.0f}% in category)</p>
                    <p>Category median likes: {ps['like_num_median']:.0f} | collections: {ps['collect_num_median']:.0f}</p>
                </div>""", unsafe_allow_html=True)

                # Mini bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Likes', 'Collections', 'Comments', 'Shares'],
                    y=[ps['like_num_median'], ps['collect_num_median'],
                       ps['comment_num_median'], ps['share_num_median']],
                    name='Category Median',
                    marker_color='#bdbdbd'
                ))
                fig.add_trace(go.Bar(
                    x=['Likes', 'Collections', 'Comments', 'Shares'],
                    y=[like_est_adj, collect_est_adj,
                       ps['comment_num_median'] * ratio,
                       ps['share_num_median'] * ratio],
                    name='Your Estimate',
                    marker_color='#ff9800'
                ))
                fig.update_layout(
                    barmode='group', height=200,
                    margin=dict(t=10, b=30),
                    legend=dict(orientation='h', y=1.15),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Counterfactual Analysis ──
        st.markdown('<div class="section-header">Counterfactual Analysis</div>', unsafe_allow_html=True)

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
            title="Expected Purchase Rate Change (if feature changed to category benchmark)",
            xaxis_title="Probability Change (%)",
            height=280, margin=dict(l=130, t=40),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: User Persona + Recommendations ──
        col_persona, col_recs = st.columns([1, 1.5])

        with col_persona:
            st.markdown('<div class="section-header">Target User Persona</div>', unsafe_allow_html=True)

            if 'user_cluster' in df.columns:
                cat_cluster_rates = df[df['category'] == category].groupby('user_cluster')['label'].mean()
                best_cluster = cat_cluster_rates.idxmax()
                best_rate = cat_cluster_rates.max()
                bp = cluster_profiles.loc[best_cluster]

                st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#ff6f00 0%,#ff8f00 100%);text-align:left;padding:1.2rem">
                    <h3>Best Segment: Cluster {best_cluster}</h3>
                    <h1 style="font-size:1.3rem">{cluster_names[best_cluster]}</h1>
                    <p style="margin-top:0.8rem;font-size:0.85rem;opacity:0.95">
                    Purchase rate: <b>{best_rate:.1%}</b><br>
                    Avg age: <b>{bp['age']:.0f}</b> |
                    Gender: <b>{'Male' if bp['gender'] > 0.5 else 'Female'}</b><br>
                    Avg spend: <b>{bp['total_spend']:,.0f}</b> |
                    User level: <b>{bp['user_level']:.1f}</b><br>
                    Freq: <b>{bp['purchase_freq']:.0f}</b> purchases |
                    <b>{int(bp['count']):,}</b> users
                    </p>
                </div>""", unsafe_allow_html=True)

                # Show all clusters comparison for this category
                fig = go.Figure()
                colors_cl = ['#667eea', '#f5576c', '#43e97b', '#f39c12']
                for ci in range(OPTIMAL_K):
                    rate = cat_cluster_rates.get(ci, 0)
                    fig.add_trace(go.Bar(
                        x=[f'C{ci}'], y=[rate * 100],
                        name=f'C{ci}: {cluster_names[ci]}',
                        marker_color=colors_cl[ci],
                        text=[f'{rate*100:.1f}%'],
                        textposition='outside'
                    ))
                fig.update_layout(
                    title=f"Purchase Rate by Cluster ({category})",
                    yaxis_title="Purchase Rate (%)",
                    height=250, margin=dict(t=40, b=20),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_recs:
            st.markdown('<div class="section-header">Actionable Recommendations</div>', unsafe_allow_html=True)

            rec_cards = generate_recommendations_html(product_params, cat_bench, feature_vec, category)
            if rec_cards:
                for card in rec_cards:
                    st.markdown(card, unsafe_allow_html=True)
            else:
                st.success("Your product features are well-aligned with category benchmarks. No major improvements needed.")

    else:
        st.info("Enter a product title and configure features above, then click **Analyze Product** to get comprehensive analysis.")


# ═══════════════════════════════════════════════
# PAGE 3: Title A/B Test
# ═══════════════════════════════════════════════
elif page == "Title A/B Test":
    st.markdown("# Title A/B Test")
    st.markdown("Compare two product titles side-by-side to see which performs better")

    category_ab = st.selectbox("Product Category", list(category_mapping.keys()), key="ab_cat")

    col_shared, _ = st.columns([1, 1])
    with col_shared:
        st.markdown("**Shared Product Features**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ab_price = st.number_input("Price", 1.0, 1000.0, 80.0, step=5.0, key="ab_price")
        with c2:
            ab_discount = st.slider("Discount", 0.0, 0.5, 0.1, step=0.01, key="ab_disc")
        with c3:
            ab_img = st.slider("Images", 0, 10, 3, key="ab_img")
        with c4:
            ab_video = st.selectbox("Video?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="ab_vid")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Title A")
        title_a = st.text_area("Enter Title A", value="", height=80, key="title_a",
                               placeholder="e.g., 新款春装女连衣裙")

    with col_b:
        st.markdown("### Title B")
        title_b = st.text_area("Enter Title B", value="", height=80, key="title_b",
                               placeholder="e.g., 2024春季新款时尚女装连衣裙 修身显瘦百搭气质")

    compare_btn = st.button("Compare Titles", type="primary", use_container_width=True)

    if compare_btn and title_a.strip() and title_b.strip():
        user_params = get_default_user_params()
        cat_bench_ab = benchmarks[category_ab]

        results_ab = []
        for label, title in [("A", title_a.strip()), ("B", title_b.strip())]:
            tl = len(title)
            raw_s, cal_s = compute_sentiment(title)
            pp = {
                'title_length': tl,
                'title_emo_score': cal_s,
                'img_count': ab_img,
                'has_video': ab_video,
                'price': ab_price,
                'discount_rate': ab_discount,
                'category': category_ab,
            }
            fv = build_feature_vector(pp, user_params, feature_cols, category_mapping)
            prob = predict_proba_safe(model, fv, feature_cols)
            d_total, d_bd = compute_domain_score(pp, cat_bench_ab, title)
            h_score, ml_s = compute_hybrid_score(prob, d_total)
            results_ab.append({
                'label': label,
                'title': title,
                'length': tl,
                'raw_sentiment': raw_s,
                'sentiment': cal_s,
                'purchase_prob': prob,
                'domain_score': d_total,
                'hybrid_score': h_score,
                'domain_breakdown': d_bd,
            })

        ra, rb = results_ab[0], results_ab[1]
        winner = "A" if ra['hybrid_score'] > rb['hybrid_score'] else "B"
        diff_hybrid = abs(ra['hybrid_score'] - rb['hybrid_score'])

        st.markdown("---")

        # Results comparison
        col_ra, col_mid, col_rb = st.columns([2, 1, 2])

        with col_ra:
            css = "ab-better" if winner == "A" else "ab-worse"
            st.markdown(f"""<div class="title-card {css}" style="min-height:280px">
                <h4>Title A {"(Winner)" if winner == "A" else ""}</h4>
                <p style="font-size:0.9rem;color:#333;margin:0.5rem 0">"{ra['title']}"</p>
                <p>Length: <b>{ra['length']}</b> chars | Sentiment: <b>{ra['sentiment']:.3f}</b></p>
                <p>ML Prob: <b>{ra['purchase_prob']:.1%}</b> | Domain: <b>{ra['domain_score']:.0f}</b></p>
                <p style="font-size:1.4rem;margin-top:0.5rem;color:#1a237e">Hybrid Score: <b>{ra['hybrid_score']:.1f}</b></p>
            </div>""", unsafe_allow_html=True)

        with col_mid:
            st.markdown(f"""<div style="text-align:center;padding-top:60px">
                <p style="font-size:2rem;font-weight:bold;color:#667eea">VS</p>
                <p style="font-size:0.9rem;color:#666">Hybrid diff: {diff_hybrid:.1f}</p>
            </div>""", unsafe_allow_html=True)

        with col_rb:
            css = "ab-better" if winner == "B" else "ab-worse"
            st.markdown(f"""<div class="title-card {css}" style="min-height:280px">
                <h4>Title B {"(Winner)" if winner == "B" else ""}</h4>
                <p style="font-size:0.9rem;color:#333;margin:0.5rem 0">"{rb['title']}"</p>
                <p>Length: <b>{rb['length']}</b> chars | Sentiment: <b>{rb['sentiment']:.3f}</b></p>
                <p>ML Prob: <b>{rb['purchase_prob']:.1%}</b> | Domain: <b>{rb['domain_score']:.0f}</b></p>
                <p style="font-size:1.4rem;margin-top:0.5rem;color:#1a237e">Hybrid Score: <b>{rb['hybrid_score']:.1f}</b></p>
            </div>""", unsafe_allow_html=True)

        # Detail comparison chart
        fig = make_subplots(rows=1, cols=4, subplot_titles=["Title Length", "Calibrated Sentiment", "Domain Score", "Hybrid Score"])

        for i, (metric, vals) in enumerate([
            ("Length", [ra['length'], rb['length']]),
            ("Sentiment", [ra['sentiment'], rb['sentiment']]),
            ("Domain", [ra['domain_score'], rb['domain_score']]),
            ("Hybrid", [ra['hybrid_score'], rb['hybrid_score']])
        ]):
            fig.add_trace(go.Bar(
                x=['A', 'B'], y=vals,
                marker_color=['#667eea', '#f5576c'],
                text=[f'{v:.1f}' for v in vals],
                textposition='outside',
                showlegend=False
            ), row=1, col=i+1)

        fig.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Domain breakdown comparison
        st.markdown('<div class="section-header">Domain Score Breakdown Comparison</div>', unsafe_allow_html=True)
        bd_keys = [k for k in ra['domain_breakdown'] if not k.startswith('_')]
        bd_label_map = {
            'title_quality': 'Title Quality', 'title_length': 'Title Length',
            'sentiment': 'Sentiment', 'images': 'Images', 'video': 'Video',
            'price': 'Price', 'discount': 'Discount',
        }
        bd_labels = [bd_label_map.get(k, k) for k in bd_keys]
        fig_bd = go.Figure()
        fig_bd.add_trace(go.Bar(
            name='Title A', y=bd_labels, x=[ra['domain_breakdown'][k] for k in bd_keys],
            orientation='h', marker_color='#667eea',
            text=[f"{ra['domain_breakdown'][k]:.0f}" for k in bd_keys], textposition='outside'
        ))
        fig_bd.add_trace(go.Bar(
            name='Title B', y=bd_labels, x=[rb['domain_breakdown'][k] for k in bd_keys],
            orientation='h', marker_color='#f5576c',
            text=[f"{rb['domain_breakdown'][k]:.0f}" for k in bd_keys], textposition='outside'
        ))
        fig_bd.update_layout(
            barmode='group', height=300, xaxis_range=[0, 115],
            margin=dict(l=110, t=10, b=20),
            legend=dict(orientation='h', y=1.15),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bd, use_container_width=True)

        # Title quality sub-detail comparison
        tq_a = ra['domain_breakdown'].get('_tq_detail', {})
        tq_b = rb['domain_breakdown'].get('_tq_detail', {})
        if tq_a and tq_b:
            st.markdown('<div class="section-header">Title Quality Detail</div>', unsafe_allow_html=True)
            tq_keys = ['char_quality', 'relevance', 'info_density']
            tq_labels = ['Char Quality', 'Category Relevance', 'Info Density']
            fig_tq = go.Figure()
            fig_tq.add_trace(go.Bar(
                name='Title A', y=tq_labels, x=[tq_a.get(k, 0) for k in tq_keys],
                orientation='h', marker_color='#667eea',
                text=[f"{tq_a.get(k, 0):.0f}" for k in tq_keys], textposition='outside'
            ))
            fig_tq.add_trace(go.Bar(
                name='Title B', y=tq_labels, x=[tq_b.get(k, 0) for k in tq_keys],
                orientation='h', marker_color='#f5576c',
                text=[f"{tq_b.get(k, 0):.0f}" for k in tq_keys], textposition='outside'
            ))
            fig_tq.update_layout(
                barmode='group', height=220, xaxis_range=[0, 115],
                margin=dict(l=130, t=10, b=20),
                legend=dict(orientation='h', y=1.15),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_tq, use_container_width=True)

    elif compare_btn:
        st.warning("Please enter both Title A and Title B.")


# ═══════════════════════════════════════════════
# PAGE 4: User Clusters
# ═══════════════════════════════════════════════
elif page == "User Clusters":
    st.markdown("# User Cluster Analysis")
    st.markdown("Understanding user segments for targeted marketing")

    if 'user_cluster' not in df.columns:
        st.warning("Run the notebook first to generate cluster assignments.")
        st.stop()

    cols = st.columns(OPTIMAL_K)
    colors_cluster = ['#667eea', '#f5576c', '#43e97b', '#f39c12']
    for i, col in enumerate(cols):
        with col:
            pr = cluster_profiles.loc[i, 'purchase_rate']
            cnt = int(cluster_profiles.loc[i, 'count'])
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,{colors_cluster[i]} 0%,{colors_cluster[i]}88 100%)">
                <h3>Cluster {i}</h3>
                <h1>{pr:.1%}</h1>
                <p style="margin:0;font-size:0.75rem;opacity:0.8">{cluster_names[i]}<br>{cnt:,} users</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Cluster Feature Profiles</div>', unsafe_allow_html=True)
        radar_features = ['age', 'user_level', 'purchase_freq', 'total_spend', 'register_days']
        cluster_means = df.groupby('user_cluster')[radar_features].mean()
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

        fig = go.Figure()
        for i in range(OPTIMAL_K):
            vals = cluster_means_norm.loc[i].tolist()
            vals.append(vals[0])
            cats = radar_features + [radar_features[0]]
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
            pr_video = df[c_mask & (df['has_video'] == 1)]['label'].mean()
            pr_no_video = df[c_mask & (df['has_video'] == 0)]['label'].mean()

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

    st.markdown('<div class="section-header">Detailed Cluster Profiles</div>', unsafe_allow_html=True)
    display_cols = ['age', 'gender', 'user_level', 'purchase_freq', 'total_spend', 'register_days', 'purchase_rate', 'count']
    display_df = cluster_profiles[display_cols].copy()
    display_df.index = [f'Cluster {i} ({cluster_names[i]})' for i in display_df.index]
    display_df['purchase_rate'] = display_df['purchase_rate'].apply(lambda x: f'{x:.1%}')
    display_df['count'] = display_df['count'].apply(lambda x: f'{x:,.0f}')
    display_df['total_spend'] = display_df['total_spend'].apply(lambda x: f'{x:,.0f}')
    st.dataframe(display_df, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 5: Category Insights
# ═══════════════════════════════════════════════
elif page == "Category Insights":
    st.markdown("# Category Insights")
    st.markdown("Benchmark analysis by product category")

    selected_cat = st.selectbox("Select Category", list(benchmarks.keys()))

    cat_data = df[df['category'] == selected_cat]
    cat_purchased = cat_data[cat_data['label'] == 1]
    cat_not_purchased = cat_data[cat_data['label'] == 0]
    cat_bench = benchmarks[selected_cat]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{len(cat_data):,}")
    with col2:
        st.metric("Purchase Rate", f"{cat_data['label'].mean():.1%}")
    with col3:
        st.metric("Avg Price", f"{cat_data['price'].mean():.1f}")
    with col4:
        st.metric("Video Rate", f"{cat_data['has_video'].mean():.1%}")

    st.markdown("---")

    st.markdown('<div class="section-header">Category Benchmarks (from purchased items)</div>', unsafe_allow_html=True)

    bench_display = pd.DataFrame({
        'Feature': SELLER_CONTROLLABLE,
        'Category Median (All)': [cat_data[f].median() for f in SELLER_CONTROLLABLE],
        'Benchmark (Purchased)': [cat_bench[f] for f in SELLER_CONTROLLABLE],
        'Category Mean (All)': [cat_data[f].mean() for f in SELLER_CONTROLLABLE],
    })
    st.dataframe(bench_display, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Feature Distributions: Purchased vs Not Purchased</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    continuous_feats = ['title_length', 'title_emo_score', 'img_count', 'price', 'discount_rate']
    for i, feat in enumerate(continuous_feats):
        with (col1 if i % 2 == 0 else col2):
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=cat_not_purchased[feat], name='Not Purchased',
                marker_color='#ff5252', opacity=0.6, histnorm='probability density'
            ))
            fig.add_trace(go.Histogram(
                x=cat_purchased[feat], name='Purchased',
                marker_color='#00c853', opacity=0.6, histnorm='probability density'
            ))
            fig.update_layout(
                title=f'{feat} Distribution',
                barmode='overlay', height=300,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.add_vline(x=cat_bench[feat], line_dash="dash", line_color="blue",
                          annotation_text=f"Benchmark: {cat_bench[feat]:.1f}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Video Impact</div>', unsafe_allow_html=True)
    video_rates = cat_data.groupby('has_video')['label'].mean()
    fig = go.Figure(go.Bar(
        x=['No Video', 'Has Video'],
        y=video_rates.values * 100,
        marker_color=['#ff5252', '#00c853'],
        text=[f'{v*100:.1f}%' for v in video_rates.values],
        textposition='outside'
    ))
    fig.update_layout(
        title=f"Purchase Rate: Video vs No Video ({selected_cat})",
        yaxis_title="Purchase Rate (%)", height=350
    )
    st.plotly_chart(fig, use_container_width=True)
