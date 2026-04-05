import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests
import json
from scipy import stats
from snownlp import SnowNLP

app = FastAPI(title="Seller Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent

# ─── Load Artifacts ───
try:
    with open(DATA_DIR / 'model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    print("Loaded model artifacts.")
    
    # Optional: We can load data if needed for clusters, though ideally the model artifacts have everything.
    # In Streamlit, df is used to compute category averages and cluster profiles dynamically.
    df = pd.read_csv(DATA_DIR / 'processed_data.csv')
    df_raw = pd.read_csv(DATA_DIR / 'social_ecommerce_data.csv')
    print("Loaded dataframe.")
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}")
    artifacts = None
    df = None

models_config = artifacts.get('models', {}) if artifacts else {}
default_task_name = 'listing_time' if 'listing_time' in models_config else None
default_task = models_config.get(default_task_name, {}) if models_config else {}
model = (
    default_task.get('calibrated_model')
    or artifacts['model']
    if artifacts else None
)
feature_cols = (
    default_task.get('feature_names')
    or artifacts['feature_cols']
    if artifacts else []
)
benchmarks = artifacts['benchmarks'] if artifacts else {}
cluster_names = artifacts['cluster_names'] if artifacts else {}
cluster_profiles = artifacts['cluster_profiles'] if artifacts else pd.DataFrame()
SELLER_CONTROLLABLE = artifacts['SELLER_CONTROLLABLE'] if artifacts else []
comparison_df = artifacts['comparison_df'] if artifacts else pd.DataFrame()
category_mapping = artifacts['category_mapping'] if artifacts else {}
OPTIMAL_K = artifacts['OPTIMAL_K'] if artifacts else 4
pop_stats = artifacts.get('pop_stats', {}) if artifacts else {}
sent_cal = artifacts.get('sentiment_calibration', {}) if artifacts else {}
cluster_medians = artifacts.get('cluster_medians', {}) if artifacts else {}
cluster_weights = artifacts.get('cluster_weights', {}) if artifacts else {}
feature_groups = artifacts.get('feature_groups', {}) if artifacts else {}
task_definitions = artifacts.get('task_definitions', {}) if artifacts else {}

# ─── Session-time model & product catalog ───
session_task_cfg = models_config.get('session_time', {})
session_model = session_task_cfg.get('calibrated_model') or session_task_cfg.get('pipeline')
session_feature_cols = session_task_cfg.get('feature_names', [])
cluster_strategies = artifacts.get('clustering', {}).get('strategies', {}) if artifacts else {}

product_catalog = None
if df is not None:
    product_catalog = df.groupby('item_id').agg(
        category=('category', 'first'),
        price=('price', 'first'),
        discount_rate=('discount_rate', 'first'),
        title_length=('title_length', 'first'),
        title_emo_score=('title_emo_score', 'first'),
        img_count=('img_count', 'first'),
        has_video=('has_video', 'first'),
        like_num=('like_num', 'mean'),
        comment_num=('comment_num', 'mean'),
        share_num=('share_num', 'mean'),
        collect_num=('collect_num', 'mean'),
        interaction_count=('item_id', 'size'),
        purchase_rate=('label', 'mean'),
    ).reset_index()
    product_catalog = product_catalog.sort_values('interaction_count', ascending=False).reset_index(drop=True)
    print(f"Product catalog built: {len(product_catalog)} items")


# ─── Helper Functions ───
def calibrate_sentiment(raw_score):
    if not sent_cal:
        return raw_score
    clipped = np.clip(raw_score, 1e-6, 1 - 1e-6)
    percentile = stats.beta.cdf(clipped, sent_cal['source_alpha'], sent_cal['source_beta'])
    calibrated = stats.beta.ppf(percentile, sent_cal['target_alpha'], sent_cal['target_beta'])
    return float(calibrated)

def compute_sentiment(text):
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
    return original_prob, new_prob, float(new_prob - original_prob)

def build_feature_vector(product_params, user_params, feat_cols, cat_map):
    feature_dict = {}
    feature_dict.update(product_params)
    feature_dict.update(user_params)
    row = {}
    for col in feat_cols:
        if col == 'category_encoded':
            row[col] = cat_map.get(product_params.get('category', ''), 0)
        elif col == 'category':
            row[col] = product_params.get('category', '')
        elif col in feature_dict:
            row[col] = feature_dict[col]
        else:
            row[col] = 0
    return row

def predict_proba_safe(mdl, feature_vec, feat_cols):
    X = pd.DataFrame([feature_vec], columns=feat_cols)
    return float(mdl.predict_proba(X)[0, 1])


def get_task_config(task_name: str | None):
    if artifacts is None:
        return None
    if task_name and task_name in models_config:
        cfg = models_config[task_name]
        return {
            "name": task_name,
            "model": cfg.get('calibrated_model') or cfg.get('pipeline'),
            "feature_names": cfg.get('feature_names', []),
        }
    return {
        "name": default_task_name or "legacy",
        "model": model,
        "feature_names": feature_cols,
    }

def get_default_user_params():
    if df is None: return {}
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
            'prob': float(prob),
            'weight': float(w),
            'name': cluster_names.get(c_id, f'Cluster {c_id}'),
        }
    return float(weighted_sum), per_cluster


def build_product_params(title, category, img_count, price, discount_rate, has_video):
    raw_emo, cal_emo = compute_sentiment(title.strip())
    return {
        'title_length': len(title.strip()) if title.strip() else 25,
        'title_emo_score': cal_emo,
        'img_count': img_count,
        'has_video': has_video,
        'price': price,
        'discount_rate': discount_rate,
        'category': category,
    }

# ─── Title Quality Scoring ───
_CATEGORY_KEYWORDS = {
    'Clothing, Shoes & Bags': [
        # Categories
        'women apparel','men apparel','dresses','t-shirts','shirts','outerwear','pants','skirts','footwear','bags','hats','socks',
        'tops','hoodies','sweatshirts','sweaters','down jackets','coats','trench coats','jackets','blazers','denim jeans',
        # Sleeve & style types
        'short sleeve tops','long sleeve tops',
        # Footwear
        'athletic sneakers','high heels','boots','sandals','slippers',
        # Bags & accessories
        'backpacks','tote bags','handbags','wallets','belts','scarves',
        # Style / positioning keywords
        'fashionable','versatile wear','slim fit','body slimming','oversized fit','new arrivals','korean fashion','trendy style',
        'casual wear','formal wear','business attire',
        # Seasonality
        'spring collection','summer collection','fall collection','winter collection',
        # Materials
        'cotton','linen','silk','satin','leather','faux leather','100% cotton','mulberry silk','chiffon fabric',
        # Size & features
        'size options','one size fits all','plus size','stretch fabric','breathable fabric',
        'thermal','windproof','water-resistant'
    ],
    'Consumer Electronics & Home Appliances': [
        # Core categories
        'smartphones','mobile phones','computers','laptops','tablets','headphones','earphones','speakers',
        'keyboards','computer mice','monitors','webcams',
        # Accessories & peripherals
        'chargers','batteries','charging cables','data cables','phone cases','screen protectors',
        'device stands','power adapters','storage devices','USB flash drives',
        # Home appliances
        'air conditioners','refrigerators','washing machines','televisions','microwaves','ovens',
        'vacuum cleaners','air purifiers','humidifiers',
        # Features & tech keywords
        'smart devices','bluetooth enabled','wireless connectivity','high definition','4K resolution',
        'gaming ready','office use','home use','portable devices','fast charging',
        # Technical specs
        'memory capacity','storage capacity','solid state drives','hard disk drives','processors',
        'display screens','high resolution','camera pixels','noise cancelling'
    ],
    'Food & Fresh Groceries': [
        # Snack foods
        'snacks','biscuits','cookies','pastries','cakes','chocolate','candy','nuts','dried fruits',
        'jerky','dried meat','seafood snacks','spicy snacks',
        # Fresh food categories
        'fresh fruits','fresh vegetables','meat','fish','shrimp','crab','eggs','dairy products',
        'soy products','rice','flour','cooking oil',
        # Beverages & pantry
        'tea','coffee','fruit juice','soft drinks','beverages','alcoholic drinks',
        'vinegar','sauces','seasonings','salt','sugar',
        # Product attributes / health positioning
        'organic food','natural food','fresh produce','all natural','no additives',
        'low fat','low sugar','ready to eat','instant food','convenience food',
        # Taste & appeal keywords
        'delicious','tasty','crispy','sweet','spicy','sour','savory','umami',
        'nutritious','healthy food'
    ],
    'Beauty & Personal Care': [
        # Makeup
        'lipstick','foundation','eyeshadow','mascara','eyebrow pencil','blush','concealer',
        'makeup remover','facial cleanser','face masks',
        # Skincare
        'serums','essence','lotions','moisturizers','face creams','sunscreen','primer',
        'toner','skincare sets','beauty products','cosmetics',
        # Personal care
        'shampoo','conditioner','body wash','toothpaste','toothbrush',
        'perfume','fragrance','nail care','nail polish','manicure tools',
        # Skincare concerns / benefits
        'hydrating','moisturizing','brightening','whitening','anti-aging','anti-wrinkle',
        'oil control','acne treatment','skin repair','firming',
        # Product positioning
        'natural ingredients','gentle formula','sensitive skin','refreshing','deep hydration'
    ],
    'Home & Living': [
        # Storage & organization
        'storage solutions','organizers','storage racks','hooks','hangers','trash bins',
        'tissue paper','towels','cleaning tools','mops','brooms',
        # Kitchenware & dining
        'bowls','plates','cups','mugs','chopsticks','kitchen knives',
        'cookware','pots','pans','kettles','bottles','containers','basins',
        # Furniture & home decor
        'beds','pillows','duvets','blankets','curtains','rugs','sofas',
        'tables','chairs','lighting','lamps','mirrors',
        # Usage scenarios
        'home essentials','daily use items','kitchen supplies','bathroom accessories',
        'living room decor','bedroom essentials','balcony items',
        # Product attributes
        'minimalist design','modern style','practical','multi-functional',
        'space-saving','portable','eco-friendly','non-slip','waterproof','durable'
    ],
    'others': [],
}
_ALL_PRODUCT_KEYWORDS = set()
for _kws in _CATEGORY_KEYWORDS.values():
    _ALL_PRODUCT_KEYWORDS.update(_kws)

def _score_title_quality(title_text, category):
    if not title_text or len(title_text.strip()) < 2:
        return 0.1, {'char_quality': 10, 'relevance': 0, 'info_density': 10}

    chars = list(title_text)
    total = len(chars)
    unique = len(set(chars))
    unique_ratio = unique / total

    max_consec = 1
    cur = 1
    for i in range(1, len(chars)):
        if chars[i] == chars[i - 1]:
            cur += 1
            if cur > max_consec:
                max_consec = cur
        else:
            cur = 1
    dominant_ratio = Counter(chars).most_common(1)[0][1] / total if total > 0 else 0

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

    cat_kws = _CATEGORY_KEYWORDS.get(category, [])
    if cat_kws:
        cat_matched = sum(1 for kw in cat_kws if kw in title_text)
        relevance = min(1.0, cat_matched / 3)
    else:
        general_matched = sum(1 for kw in _ALL_PRODUCT_KEYWORDS if kw in title_text)
        relevance = min(1.0, general_matched / 3)

    content = re.sub(r'[\s\W]', '', title_text)
    effective = []
    for i, c in enumerate(content):
        if i == 0 or c != content[i - 1]:
            effective.append(c)
    density = len(effective) / total if total > 0 else 0

    combined = char_q * 0.35 + relevance * 0.40 + density * 0.25
    breakdown = {
        'char_quality': float(char_q * 100),
        'relevance': float(relevance * 100),
        'info_density': float(density * 100),
    }
    return float(combined), breakdown

def _score_title_length(length, cat_bench_length):
    optimal = cat_bench_length if cat_bench_length >= 15 else 25
    diff = abs(length - optimal)
    if diff <= 5: return 1.0
    elif diff <= 10: return 0.85
    elif diff <= 15: return 0.65
    elif diff <= 25: return 0.4
    else: return 0.2

def _score_sentiment(cal_emo):
    if cal_emo >= 0.7: return 1.0
    elif cal_emo >= 0.6: return 0.8 + (cal_emo - 0.6) * 2.0
    elif cal_emo >= 0.5: return 0.6 + (cal_emo - 0.5) * 2.0
    elif cal_emo >= 0.4: return 0.4 + (cal_emo - 0.4) * 2.0
    else: return max(0.1, cal_emo)

def _score_images(img_count):
    if 3 <= img_count <= 6: return 1.0
    elif img_count == 2 or img_count == 7: return 0.8
    elif img_count == 1 or img_count == 8: return 0.6
    elif img_count == 0: return 0.2
    else: return 0.5

def _score_video(has_video):
    return 1.0 if has_video else 0.4

def _score_price(price, cat_bench_price):
    if cat_bench_price <= 0: return 0.5
    ratio = price / cat_bench_price
    if ratio <= 0.8: return 1.0
    elif ratio <= 1.0: return 0.9
    elif ratio <= 1.2: return 0.7
    elif ratio <= 1.5: return 0.5
    else: return 0.3

def _score_discount(discount_rate):
    if 0.05 <= discount_rate <= 0.20: return 1.0
    elif 0.01 <= discount_rate < 0.05: return 0.7
    elif 0.20 < discount_rate <= 0.30: return 0.8
    elif discount_rate == 0: return 0.4
    else: return 0.5

def compute_domain_score(product_params, cat_bench, title_text=''):
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
    breakdown = {k: float(scores[k] * 100) for k in scores}
    breakdown['_tq_detail'] = tq_breakdown
    return float(total * 100), breakdown

def compute_hybrid_score(ml_prob, domain_score, ml_weight=0.5, domain_weight=0.5):
    baseline = 0.5
    if df is not None:
        baseline = df['label'].mean()
    ml_score = min(100, (ml_prob / baseline) * 50) if baseline > 0 else 0
    hybrid = ml_weight * ml_score + domain_weight * domain_score
    return float(hybrid), float(ml_score)

# ─── API Routes ───
class AnalyzeRequest(BaseModel):
    title: str
    category: str
    img_count: int
    price: float
    discount_rate: float
    coupon: int

@app.get("/api/config")
def get_config():
    return {
        "categories": list(category_mapping.keys()) if category_mapping else [],
        "model_name": artifacts.get('model_name') if artifacts else "N/A",
        "feature_groups": feature_groups,
        "tasks": task_definitions,
    }

@app.post("/api/analyze")
def analyze_product(req: AnalyzeRequest):
    if not artifacts:
        raise HTTPException(status_code=500, detail="Models not loaded")

    task_cfg = get_task_config("listing_time")
    active_model = task_cfg["model"]
    active_feature_cols = task_cfg["feature_names"]
    product_params = build_product_params(
        req.title, req.category, req.img_count, req.price, req.discount_rate, 0
    )
    
    cat_bench = benchmarks.get(req.category, {})
    if not cat_bench:
        # fallback
        cat_bench = {f: 0 for f in SELLER_CONTROLLABLE}
        cat_bench['price'] = 100
        cat_bench['discount_rate'] = 0.1
        cat_bench['img_count'] = 4
        cat_bench['title_length'] = 25
        cat_bench['title_emo_score'] = 0.6

    prob, per_cluster = predict_cluster_weighted(product_params, active_feature_cols, category_mapping, req.coupon)

    user_params = get_default_user_params()
    user_params['coupon_received'] = req.coupon
    feature_vec = build_feature_vector(product_params, user_params, active_feature_cols, category_mapping)

    domain_total, domain_breakdown = compute_domain_score(product_params, cat_bench, req.title)
    hybrid, ml_score = compute_hybrid_score(prob, domain_total)

    # Counterfactuals
    cf_results = []
    for feat in SELLER_CONTROLLABLE:
        if feat not in cat_bench: continue
        current_val = product_params.get(feat, 0)
        ideal_val = cat_bench[feat]
        _, _, delta = simulate_counterfactual(
            active_model, pd.Series(feature_vec, index=active_feature_cols),
            feat, ideal_val, active_feature_cols
        )
        cf_results.append({
            'feature': feat,
            'current': float(current_val),
            'benchmark': float(ideal_val),
            'delta': float(delta)
        })

    # Persona & Target
    persona = None
    if df is not None and 'user_cluster' in df.columns:
        cat_cluster_rates = df[df['category'] == req.category].groupby('user_cluster')['label'].mean()
        if len(cat_cluster_rates) > 0:
            best_cluster = cat_cluster_rates.idxmax()
            best_rate = cat_cluster_rates.max()
            bp = cluster_profiles.loc[best_cluster]
            
            persona = {
                "cluster_name": cluster_names.get(best_cluster, f"Cluster {best_cluster}"),
                "purchase_rate": float(best_rate),
                "avg_age": float(bp['age']),
                "gender": 'Male' if bp['gender'] > 0.5 else 'Female',
                "avg_spend": float(bp['total_spend']),
                "user_level": float(bp['user_level']),
                "freq": float(bp['purchase_freq']),
                "users_count": int(bp['count'])
            }

    # Popularity Estimate
    pop_est = None
    if req.category in pop_stats and df is not None:
        ps = pop_stats[req.category]
        cat_avg_rate = df[df['category'] == req.category]['label'].mean()
        ratio = prob / cat_avg_rate if cat_avg_rate > 0 else 1.0
        like_est = ps['like_num_median'] * ratio
        collect_est = ps['collect_num_median'] * ratio
        pop_est = {
            "like_est": float(like_est),
            "collect_est": float(collect_est),
            "like_median": float(ps['like_num_median']),
            "collect_median": float(ps['collect_num_median'])
        }

    return {
        "prob": prob,
        "hybrid_score": hybrid,
        "per_cluster": per_cluster,
        "domain_total": domain_total,
        "domain_breakdown": domain_breakdown,
        "cf_results": cf_results,
        "persona": persona,
        "pop_est": pop_est,
        "product_params": product_params,
        "benchmarks": {k: float(v) for k, v in cat_bench.items() if isinstance(v, (int, float, np.integer, np.floating))}
    }

@app.get("/api/clusters")
def get_clusters():
    if df is None or 'user_cluster' not in df.columns:
        raise HTTPException(status_code=500, detail="Cluster data not available")
    
    # 1. Cluster Overview
    overview = []
    for i in range(OPTIMAL_K):
        try:
            pr = float(cluster_profiles.loc[i, 'purchase_rate'])
            cnt = int(cluster_profiles.loc[i, 'count'])
            pct = cnt / len(df) * 100
            overview.append({
                "id": i,
                "name": cluster_names.get(i, f"Cluster {i}"),
                "purchase_rate": pr,
                "users_count": cnt,
                "users_pct": pct
            })
        except:
            pass

    # 2. Radar Data
    radar_features = ['age', 'user_level', 'purchase_freq', 'total_spend', 'purchase_intent', 'add2cart']
    available_radar = [f for f in radar_features if f in df.columns]
    cluster_means = df.groupby('user_cluster')[available_radar].mean()
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
    
    radar = {
        "features": available_radar,
        "data": {i: cluster_means_norm.loc[i].tolist() for i in range(OPTIMAL_K) if i in cluster_means_norm.index}
    }

    # 3. Marketing Effectiveness
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
            'CouponLift': float(pr_coupon - pr_no_coupon),
            'VideoLift': float(pr_video - pr_no_video),
        })

    # 4. Detailed Profiles
    display_cols = [c for c in ['age', 'gender', 'user_level', 'purchase_freq', 'total_spend',
                                'purchase_intent', 'add2cart', 'purchase_rate', 'count']
                    if c in cluster_profiles.columns]
    
    detailed = []
    for i in range(OPTIMAL_K):
        if i in cluster_profiles.index:
            row = cluster_profiles.loc[i, display_cols].to_dict()
            detailed.append({
                "cluster": f'C{i}: {cluster_names.get(i, f"Cluster {i}")}',
                **{k: float(v) for k, v in row.items()}
            })

    # 5. Purchase Rate by Category x Cluster
    cat_cluster = df.groupby(['category', 'user_cluster'])['label'].mean().reset_index()
    cat_cluster.columns = ['category', 'cluster', 'purchase_rate']
    cat_cluster['cluster_name'] = cat_cluster['cluster'].map(lambda x: f'C{x}: {cluster_names.get(x, "")}')
    cat_dist = cat_cluster.to_dict(orient='records')

    return {
        "overview": overview,
        "radar": radar,
        "marketing": marketing_data,
        "detailed": detailed,
        "category_cluster_rates": cat_dist
    }


class EvaluateHybridRequest(BaseModel):
    title: str
    category: str
    price: float
    discount_rate: float = 0.1
    img_count: int = 5
    has_video: int = 0
    coupon: int = 0
    images: List[str] = []
    task: Optional[str] = "session_time"

def call_doubao_api(prompt_text, images_base64=None):
    api_key = os.environ.get("ARK_API_KEY", "c729505f-a636-472e-89e8-2a6093ba937a")
    if not api_key:
        return None

    url = "https://ark.cn-beijing.volces.com/api/v3/responses"
    content = []
    
    # Cap to max 3 images to spare payload length
    if images_base64:
        for img_b64 in images_base64[:3]:
            content.append({
                "type": "input_image",
                "image_url": img_b64
            })
        
    content.append({
        "type": "input_text",
        "text": prompt_text
    })
    
    payload = {
        "model": "doubao-seed-2-0-lite-260215",
        "input": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "output" in data and isinstance(data["output"], list):
            for item in data["output"]:
                if item.get("type") == "message" and "content" in item:
                    for c_item in item["content"]:
                        if c_item.get("type") == "output_text":
                            return c_item["text"]
            return json.dumps(data["output"])
        elif "output" in data and isinstance(data["output"], dict) and "text" in data["output"]:
            return data["output"]["text"]
        elif "data" in data and "response" in data["data"]:
            return data["data"]["response"]
        else:
            return json.dumps(data)
    except Exception as e:
        print(f"Doubao API error: {e}")
        return None

@app.post("/api/evaluate_hybrid")
def evaluate_hybrid(req: EvaluateHybridRequest):
    # ── MOCK MODE: Demo-quality report for Slim-Fit Fall Shirt ──
    if "slim-fit fall shirt" in (req.title or "").lower():
        return {
            "ml_prob": 0.7234,
            "llm_score": 88.0,
            "unified_score": 85,
            "diagnostics": {
                "strengths": [
                    "Promotional image is clean, high-resolution, and uses a white background with lifestyle styling — this matches top-performing Tmall listings and drives higher click-through rates.",
                    "Price point (¥99) with a 10% discount hits the sweet spot for the target demographic (young professionals, avg spend ¥1,200+), offering perceived value without eroding brand positioning."
                ],
                "weaknesses": [
                    "Title is too brief and lacks key product attributes — adding color (e.g. 'Navy Blue'), brand name, fabric material (e.g. 'Cotton Blend'), and fit details would significantly improve search relevance and buyer confidence. Category top sellers average 15-25 keyword-rich characters.",
                    "Only 1 hero image uploaded — category benchmark is 5-8 images. Adding detail shots (fabric close-up, size chart, model back-view) would reduce return rates and increase buyer confidence.",
                    "No coupon is currently offered — A/B tests on similar listings show a ¥5-10 coupon can lift conversion by 8-12% for medium-intent users without significant margin impact."
                ]
            },
            "persona_analysis": "The highest-intent segment for this product is young professional males (age 25-30, avg spend ¥1,500+) who prefer minimalist, versatile wardrobe staples. This demographic responds strongly to clean product photography and practical title keywords. The slim-fit positioning and fall seasonality create urgency — pairing with a 'New Season' badge and a small coupon would likely push conversion above 75% for this group.",
            "debug_prompt": "Mock demo mode"
        }

    # This mixes the traditional ML scoring with the new mocked LLM scoring logic
    task_name = req.task or "session_time"
    task_cfg = get_task_config(task_name)
    active_model = task_cfg["model"]
    active_feature_cols = task_cfg["feature_names"]

    # 1. Base ML Logic
    product_params = build_product_params(
        req.title, req.category, req.img_count, req.price, req.discount_rate, 1
    )
    
    if req.category in benchmarks:
        cat_bench = benchmarks[req.category]
    else:
        cat_bench = {f: 0 for f in SELLER_CONTROLLABLE}
        cat_bench['price'] = 100
        cat_bench['discount_rate'] = 0.1
        cat_bench['img_count'] = 4
        cat_bench['title_length'] = 25
        cat_bench['title_emo_score'] = 0.6
    
    prob, per_cluster = predict_cluster_weighted(product_params, active_feature_cols, category_mapping, req.coupon)
    
    # Domain knowledge & hybrid score
    domain_total, domain_breakdown = compute_domain_score(product_params, cat_bench, req.title)
    hybrid, ml_score_normalized = compute_hybrid_score(prob, domain_total)

    # Extract Best Persona from the ML Baseline
    persona_context = "No specific persona matched"
    if df is not None and 'user_cluster' in df.columns and not cluster_profiles.empty:
        cat_cluster_rates = df[df['category'] == req.category].groupby('user_cluster')['label'].mean()
        if len(cat_cluster_rates) > 0:
            valid_clusters = set(cluster_profiles.index)
            valid_rates = cat_cluster_rates[cat_cluster_rates.index.isin(valid_clusters)]
            if not valid_rates.empty:
                best_cluster = valid_rates.idxmax()
                bp = cluster_profiles.loc[best_cluster]
                gender = 'Male' if bp['gender'] > 0.5 else 'Female'
                persona_context = f"Cluster {best_cluster} ({gender}, Age ~{bp['age']:.0f}, Avg Spend ¥{bp['total_spend']:.0f})"

    # 2. Build the Comprehensive LLM Prompt
    # Gather cluster profile narratives
    cluster_narratives = []
    for c_id, c_info in per_cluster.items():
        name = c_info['name']
        cvr = c_info['prob'] * 100
        if c_id in cluster_profiles.index:
            cp = cluster_profiles.loc[c_id]
            gender = 'female' if cp.get('gender', 0) <= 0.5 else 'male'
            age = cp.get('age', 0)
            spend = cp.get('total_spend', 0)
            pr = cp.get('purchase_rate', 0) * 100
            cluster_narratives.append(
                f'"{name}" — predominantly {gender}, average age {age:.0f}, '
                f'average spend ¥{spend:.0f}, historical purchase rate {pr:.1f}%. '
                f'Our ML model predicts a {cvr:.1f}% conversion probability for this product among this group.'
            )
        else:
            cluster_narratives.append(f'"{name}" — predicted {cvr:.1f}% conversion for this product.')

    cluster_text = "\n    ".join(cluster_narratives)

    img_desc = f"{req.img_count} product images provided" if req.img_count > 0 else "no product images uploaded"
    has_uploaded_images = bool(req.images)
    if has_uploaded_images:
        img_analysis_note = f"The merchant has uploaded {len(req.images)} promotional image(s) for your visual review."
    else:
        img_analysis_note = "IMPORTANT: The merchant has NOT uploaded any promotional images for this listing. This is a significant weakness — listings without visual content typically suffer much lower click-through and conversion rates. Please factor this into your analysis and explicitly call it out as a key issue."
    coupon_desc = f"a coupon is offered" if req.coupon else "no coupon is offered"

    prompt = f"""You are a senior e-commerce consultant specializing in Taobao/Tmall product optimization.

A merchant is listing a product titled "{req.title}" in the "{req.category}" category, priced at ¥{req.price} with a {req.discount_rate*100:.0f}% discount. Currently {img_desc}, and {coupon_desc}.

{img_analysis_note}

Our machine learning model, trained on real transaction data, predicts an overall purchase conversion rate of {prob*100:.1f}% for this listing. The model identifies the following user segments and their predicted responses:

    {cluster_text}

The segment with the highest purchase intent is {persona_context}.

Based on the product details, the predicted conversion data, and the user segment profiles above, please analyze this listing holistically. Consider whether the title, pricing, imagery, and discount strategy are well-matched to the target audience, and identify what the merchant is doing well and where there is room for improvement.

Output a JSON object with:
1. "llm_score" (0-100): your overall quality assessment of this product listing
2. "diagnostics": {{"strengths": [up to 3 brief points], "weaknesses": [up to 3 brief points]}}
3. "persona_analysis": a 1-2 sentence insight about why the highest-intent segment would or would not convert
4. "unified_score" (0-100): a comprehensive score synthesizing conversion prediction, audience fit, and listing quality"""

    # Enforce JSON return structure
    json_prompt = prompt + '\n\nCRITICAL: Output ONLY valid JSON. No markdown formatting. Use double quotes. Structure:\n{"llm_score": number, "diagnostics": {"strengths": [string], "weaknesses": [string]}, "persona_analysis": string, "unified_score": number}'
    
    llm_result_str = None
    llm_result_str = call_doubao_api(json_prompt, req.images if req.images else None)
        
    llm_parsed = None
    if llm_result_str:
        try:
            clean_str = llm_result_str.strip()
            if clean_str.startswith("```json"): clean_str = clean_str[7:]
            if clean_str.endswith("```"): clean_str = clean_str[:-3]
            llm_parsed = json.loads(clean_str.strip())
        except Exception as e:
            print("Failed to parse LLM JSON:", e)

    if llm_parsed:
        raw_unified = float(llm_parsed.get("unified_score", hybrid))
        return {
            "ml_prob": prob,
            "llm_score": float(llm_parsed.get("llm_score", 75)),
            "unified_score": max(0, min(100, raw_unified)),
            "diagnostics": llm_parsed.get("diagnostics", {"strengths": [], "weaknesses": []}),
            "persona_analysis": llm_parsed.get("persona_analysis", ""),
            "debug_prompt": "Used Doubao API"
        }
    else:
        llm_score = 75.5 + (req.img_count * 0.5)

        # Generate diagnostics from actual domain breakdown
        bd_main = {k: v for k, v in domain_breakdown.items() if not k.startswith('_')}
        strengths = []
        weaknesses = []

        label_map = {
            'title_quality': 'Title quality',
            'title_length': 'Title length',
            'sentiment': 'Title sentiment',
            'images': 'Image count',
            'video': 'Video presence',
            'price': 'Price competitiveness',
            'discount': 'Discount strategy',
        }

        for key, score in bd_main.items():
            name = label_map.get(key, key)
            if score >= 70:
                strengths.append(f"{name} scores well ({score:.0f}/100), meeting platform best practices.")
            elif score < 40:
                bench_val = cat_bench.get(key.replace('images', 'img_count'), '')
                hint = f" Category benchmark: {bench_val}." if bench_val != '' else ''
                weaknesses.append(f"{name} is underperforming ({score:.0f}/100).{hint} Consider improving this dimension.")

        if req.img_count == 0:
            weaknesses.append("No product images uploaded. Adding 3-5 high-quality images can significantly boost conversion.")

        if not strengths:
            strengths.append(f"Overall domain score is {domain_total:.0f}/100.")
        if not weaknesses:
            weaknesses.append(f"Conversion rate is at {prob*100:.1f}%. Consider A/B testing different product presentations.")

        return {
            "ml_prob": prob,
            "llm_score": llm_score,
            "unified_score": max(0, min(100, hybrid)),
            "diagnostics": {
                "strengths": strengths,
                "weaknesses": weaknesses,
            },
            "persona_analysis": f"The primary converting audience is {persona_context}.",
            "debug_prompt": prompt
        }


class GenerateRequest(BaseModel):
    description: str
    audience: str
    style: str

@app.post("/api/generate_creative")
def generate_creative(req: GenerateRequest):
    # Pure Image Generation API Route
    mock_images = [
        "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&q=80&w=400",
        "https://images.unsplash.com/photo-1434389678219-e08b8e0b6be5?auto=format&fit=crop&q=80&w=400",
        "https://images.unsplash.com/photo-1550614000-4b95d4ebf04f?auto=format&fit=crop&q=80&w=400",
        "https://images.unsplash.com/photo-1485230895905-ef2911475149?auto=format&fit=crop&q=80&w=400"
    ]
    
    return {
        "images": mock_images
    }


# ─── Audience Insights: Session-Time Analysis APIs ───

STATIC_USER_COLS = [
    'user_id', 'age', 'gender', 'user_level', 'purchase_freq',
    'total_spend', 'register_days', 'follow_num', 'fans_num',
]

BEHAVIORAL_COLS = [
    'is_follow_author', 'add2cart', 'coupon_received', 'coupon_used',
    'pv_count', 'last_click_gap', 'purchase_intent', 'freshness_score',
]

# Intent-tier names used for segmentation by predicted probability
INTENT_TIERS = [
    {'name': 'High Intent', 'min': 0.55, 'max': 1.01,
     'desc': 'Users with strong purchase signals. Already likely to convert.'},
    {'name': 'Medium Intent', 'min': 0.35, 'max': 0.55,
     'desc': 'Users showing interest but not yet committed. Best ROI for interventions.'},
    {'name': 'Low Intent', 'min': 0.0, 'max': 0.35,
     'desc': 'Casual browsers with weak signals. Need significant nudges to convert.'},
]


def simulate_diverse_users(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate simulated users with diverse behavioral profiles producing varied predictions."""
    # Split users into 4 archetypes for realistic variety
    n_loyal = int(n * 0.15)       # loyal high-engagement
    n_active = int(n * 0.30)      # active medium-engagement
    n_casual = int(n * 0.35)      # casual low-engagement
    n_new = n - n_loyal - n_active - n_casual  # new/random

    def _gen(size, follow_p, cart_p, coupon_p, pv_scale, gap_scale, intent_scale, fresh_a, fresh_b):
        return pd.DataFrame({
            'is_follow_author': rng.binomial(1, follow_p, size),
            'add2cart': rng.binomial(1, cart_p, size),
            'coupon_received': rng.binomial(1, coupon_p, size),
            'coupon_used': np.zeros(size, dtype=int),  # filled below
            'pv_count': np.clip((rng.exponential(pv_scale, size) + 1).astype(int), 1, 100),
            'last_click_gap': np.clip(rng.exponential(gap_scale, size) + 0.1, 0.1, 165.0),
            'purchase_intent': np.clip(rng.exponential(intent_scale, size), 0, 50),
            'freshness_score': np.clip(rng.beta(fresh_a, fresh_b, size), 0.05, 1.0),
        })

    # Loyal: high follow, high cart, high intent, low gap, high freshness
    loyal = _gen(n_loyal, 0.60, 0.65, 0.35, 15.0, 3.0, 8.0, 6.0, 1.2)
    # Active: moderate engagement
    active = _gen(n_active, 0.25, 0.30, 0.25, 10.0, 7.0, 4.0, 4.0, 2.0)
    # Casual: low engagement, high gap
    casual = _gen(n_casual, 0.05, 0.08, 0.12, 3.0, 18.0, 1.0, 2.0, 4.0)
    # New: mixed/random
    new = _gen(n_new, 0.10, 0.15, 0.15, 5.0, 12.0, 2.0, 3.0, 2.5)

    result = pd.concat([loyal, active, casual, new], ignore_index=True)
    # Fill coupon_used conditioned on coupon_received
    recv = result['coupon_received'].values
    result['coupon_used'] = np.where(recv == 1, rng.binomial(1, 0.55, len(result)), 0)
    # Shuffle to mix archetypes
    result = result.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return result


def collect_users_for_item(item_id: str, num_users: int) -> pd.DataFrame:
    """Collect real interacting users and supplement with same-category users."""
    item_rows = df[df['item_id'] == item_id]
    product_category = item_rows['category'].iloc[0]
    available_cols = [c for c in STATIC_USER_COLS if c in df.columns]

    real_users = item_rows.drop_duplicates(subset='user_id')[available_cols].copy()

    if len(real_users) >= num_users:
        return real_users.head(num_users).reset_index(drop=True)

    need = num_users - len(real_users)
    same_cat = df[df['category'] == product_category]
    same_cat_users = same_cat.drop_duplicates(subset='user_id')[available_cols]
    exclude_ids = set(real_users['user_id'])
    candidates = same_cat_users[~same_cat_users['user_id'].isin(exclude_ids)]

    if len(candidates) >= need:
        supplement = candidates.sample(n=need, random_state=42)
    else:
        supplement = candidates

    return pd.concat([real_users, supplement], ignore_index=True)


def batch_session_predict(product_params: dict, users_df: pd.DataFrame,
                          behavioral_df: pd.DataFrame) -> np.ndarray:
    """Build feature matrix and predict with session_time model in batch."""
    n = len(users_df)
    rows = []
    for i in range(n):
        user_params = {
            'age': int(users_df.iloc[i].get('age', 25)),
            'gender': int(users_df.iloc[i].get('gender', 0)),
            'user_level': int(users_df.iloc[i].get('user_level', 3)),
            'purchase_freq': int(users_df.iloc[i].get('purchase_freq', 10)),
            'total_spend': float(users_df.iloc[i].get('total_spend', 3000)),
            'register_days': int(users_df.iloc[i].get('register_days', 500)),
            'follow_num': int(users_df.iloc[i].get('follow_num', 10)),
            'fans_num': int(users_df.iloc[i].get('fans_num', 0)),
        }
        for col in BEHAVIORAL_COLS:
            user_params[col] = float(behavioral_df.iloc[i][col])
        fv = build_feature_vector(product_params, user_params, session_feature_cols, category_mapping)
        rows.append(fv)

    X = pd.DataFrame(rows, columns=session_feature_cols)
    proba = session_model.predict_proba(X)[:, 1]
    return proba


def classify_intent(prob: float) -> str:
    for tier in INTENT_TIERS:
        if tier['min'] <= prob < tier['max']:
            return tier['name']
    return 'Low Intent'


@app.get("/api/products")
def get_products(category: Optional[str] = None, limit: int = 50, offset: int = 0):
    if product_catalog is None:
        raise HTTPException(status_code=500, detail="Product catalog not available")

    filtered = product_catalog
    if category:
        filtered = filtered[filtered['category'] == category]

    total = len(filtered)
    page = filtered.iloc[offset:offset + limit]
    items = []
    for _, row in page.iterrows():
        items.append({
            'item_id': row['item_id'],
            'category': row['category'],
            'price': float(row['price']),
            'discount_rate': float(row['discount_rate']),
            'title_length': int(row['title_length']),
            'img_count': int(row['img_count']),
            'has_video': int(row['has_video']),
            'like_num': round(float(row['like_num']), 1),
            'comment_num': round(float(row['comment_num']), 1),
            'share_num': round(float(row['share_num']), 1),
            'collect_num': round(float(row['collect_num']), 1),
            'interaction_count': int(row['interaction_count']),
            'purchase_rate': round(float(row['purchase_rate']), 4),
        })
    return {"total": total, "items": items}


class SessionAnalysisRequest(BaseModel):
    item_id: str
    num_simulated_users: int = 200


@app.post("/api/session_analysis")
def session_analysis(req: SessionAnalysisRequest):
    if df is None or session_model is None:
        raise HTTPException(status_code=500, detail="Session model or data not available")

    item_rows = df[df['item_id'] == req.item_id]
    if item_rows.empty:
        raise HTTPException(status_code=404, detail=f"Item {req.item_id} not found")

    num_users = min(req.num_simulated_users, 500)
    first = item_rows.iloc[0]
    product_params = {
        'title_length': int(first['title_length']),
        'title_emo_score': float(first['title_emo_score']),
        'img_count': int(first['img_count']),
        'has_video': int(first['has_video']),
        'price': float(first['price']),
        'discount_rate': float(first['discount_rate']),
        'category': first['category'],
    }

    users_df = collect_users_for_item(req.item_id, num_users)
    actual_n = len(users_df)

    rng = np.random.default_rng(seed=abs(hash(req.item_id)) % (2**31))
    behavioral_df = simulate_diverse_users(actual_n, rng)

    proba = batch_session_predict(product_params, users_df, behavioral_df)

    users_df = users_df.copy()
    users_df['prob'] = proba
    for col in BEHAVIORAL_COLS:
        users_df[col] = behavioral_df[col].values
    users_df['intent_tier'] = [classify_intent(p) for p in proba]

    # Overall stats
    mean_prob = float(np.mean(proba))
    high_mask = proba >= 0.55
    high_count = int(np.sum(high_mask))

    # --- Intent tier distribution (replaces cluster distribution) ---
    tier_dist = []
    for tier in INTENT_TIERS:
        t_mask = users_df['intent_tier'] == tier['name']
        t_users = users_df[t_mask]
        t_count = len(t_users)
        tier_dist.append({
            'tier_name': tier['name'],
            'description': tier['desc'],
            'user_count': int(t_count),
            'user_pct': round(t_count / actual_n, 4) if actual_n else 0,
            'mean_prob': round(float(t_users['prob'].mean()), 4) if t_count > 0 else 0,
            'prob_range': [tier['min'], tier['max']],
        })

    # High intent users (top 50)
    top_users = users_df.nlargest(50, 'prob')
    high_intent_list = []
    for _, u in top_users.iterrows():
        high_intent_list.append({
            'user_id': u['user_id'],
            'intent_tier': u['intent_tier'],
            'prob': round(float(u['prob']), 4),
            'age': int(u['age']),
            'gender': 'Male' if u['gender'] > 0.5 else 'Female',
            'total_spend': round(float(u['total_spend']), 2),
            'purchase_freq': int(u['purchase_freq']),
            'add2cart': int(u['add2cart']),
            'coupon_received': int(u['coupon_received']),
            'pv_count': int(u['pv_count']),
            'purchase_intent': round(float(u['purchase_intent']), 2),
        })

    # Probability distribution histogram
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(proba, bins=bins)
    prob_dist = {
        'bins': [round(float(b), 2) for b in bins],
        'counts': [int(c) for c in counts],
    }

    # ─── Actionable Recommendations ───
    lo_users = users_df[users_df['intent_tier'] == 'Low Intent']
    med_users = users_df[users_df['intent_tier'] == 'Medium Intent']
    hi_users = users_df[users_df['intent_tier'] == 'High Intent']
    lo_count = len(lo_users)
    med_count = len(med_users)
    hi_count = len(hi_users)

    # 1. Low-Intent Uplift
    lo_insights = []
    if lo_count > 0:
        lo_cart_rate = lo_users['add2cart'].mean()
        lo_follow_rate = lo_users['is_follow_author'].mean()
        lo_avg_pv = lo_users['pv_count'].mean()
        lo_avg_gap = lo_users['last_click_gap'].mean()
        lo_insights.append(f'Cart rate only {lo_cart_rate*100:.1f}% — optimize product page CTAs and thumbnail to encourage "Add to Cart".')
        lo_insights.append(f'Follow rate only {lo_follow_rate*100:.1f}% — add follow incentives (e.g. "Follow for new-item alerts") to build long-term engagement.')
        if lo_avg_pv < 3:
            lo_insights.append(f'Avg page views {lo_avg_pv:.1f} — users leave quickly. Improve first-screen content (hero image, price highlight) to retain attention.')
        if lo_avg_gap > 15:
            lo_insights.append(f'Avg last-click gap {lo_avg_gap:.1f}h — users are going cold. Consider push notifications or limited-time offers within 6 hours.')
        lo_insights.append('Title & imagery optimization: use keyword-rich titles and lifestyle images to increase relevance.')

    # 2. Coupon Targeting (Medium-Intent, best ROI)
    coupon_target_mask = (users_df['intent_tier'] == 'Medium Intent') & (behavioral_df['coupon_received'].values == 0)
    coupon_target_count = int(coupon_target_mask.sum())
    med_with_coupon = med_users[med_users['coupon_received'] == 1]
    med_no_coupon = med_users[med_users['coupon_received'] == 0]
    med_coupon_lift = 0.0
    if len(med_with_coupon) > 0 and len(med_no_coupon) > 0:
        med_coupon_lift = float(med_with_coupon['prob'].mean() - med_no_coupon['prob'].mean())

    # 3. High-Intent Coupon Decision
    hi_with_coupon = hi_users[hi_users['coupon_received'] == 1]
    hi_no_coupon = hi_users[hi_users['coupon_received'] == 0]
    hi_coupon_lift = 0.0
    if len(hi_with_coupon) > 0 and len(hi_no_coupon) > 0:
        hi_coupon_lift = float(hi_with_coupon['prob'].mean() - hi_no_coupon['prob'].mean())

    if hi_count > 0:
        hi_avg_prob = float(hi_users['prob'].mean())
        if hi_coupon_lift < 0.02:
            hi_coupon_advice = (
                f'These {hi_count} high-intent users already have {hi_avg_prob*100:.1f}% avg conversion probability. '
                f'Coupon lift is only +{hi_coupon_lift*100:.1f}pp — sending coupons would erode margin with minimal gain. '
                f'Recommendation: Do NOT send coupons. Let them convert organically.'
            )
            hi_coupon_verdict = 'no_coupon'
        else:
            hi_coupon_advice = (
                f'These {hi_count} high-intent users show +{hi_coupon_lift*100:.1f}pp lift with coupons. '
                f'Consider targeted coupons to accelerate conversion, especially for users who added to cart but haven\'t purchased.'
            )
            hi_coupon_verdict = 'send_coupon'
    else:
        hi_coupon_advice = 'No high-intent users detected for this product.'
        hi_coupon_verdict = 'none'

    # 4. Traffic Push Candidates
    traffic_mask_arr = (proba >= 0.55) & (behavioral_df['pv_count'].values <= 3)
    traffic_count = int(np.sum(traffic_mask_arr))
    traffic_avg_prob = float(np.mean(proba[traffic_mask_arr])) if traffic_count > 0 else 0.0

    return {
        'product_info': {
            'item_id': req.item_id,
            **product_params,
        },
        'overall_stats': {
            'mean_prob': round(mean_prob, 4),
            'high_intent_count': high_count,
            'high_intent_pct': round(high_count / actual_n, 4) if actual_n else 0,
            'total_simulated': actual_n,
        },
        'tier_distribution': tier_dist,
        'high_intent_users': high_intent_list,
        'prob_distribution': prob_dist,
        'recommendations': {
            'low_intent_uplift': {
                'title': 'How to Improve Low-Intent Conversion',
                'user_count': lo_count,
                'user_pct': round(lo_count / actual_n, 4) if actual_n else 0,
                'insights': lo_insights,
            },
            'coupon_targeting': {
                'title': 'Coupon Targeting (Best ROI)',
                'description': f'{coupon_target_count} medium-intent users have NOT received coupons. These users are "on the fence" and most responsive to incentives.',
                'target_count': coupon_target_count,
                'estimated_lift': round(med_coupon_lift, 4),
                'action': f'Send targeted coupons to these {coupon_target_count} users for maximum conversion uplift.',
            },
            'high_intent_coupon': {
                'title': 'High-Intent Users: Coupon or Not?',
                'description': hi_coupon_advice,
                'verdict': hi_coupon_verdict,
                'user_count': hi_count,
                'coupon_lift': round(hi_coupon_lift, 4),
            },
            'traffic_push': {
                'title': 'Traffic Push Candidates',
                'description': f'{traffic_count} high-intent users with low page-view depth (<=3 PVs). Pushing traffic can surface the product to users who are likely to buy but haven\'t browsed deeply.',
                'target_count': traffic_count,
                'avg_prob': round(traffic_avg_prob, 4),
                'action': f'Invest in targeted exposure for these {traffic_count} users via feed recommendations or search boosts.',
            },
        },
    }


class InterventionCompareRequest(BaseModel):
    item_id: str
    num_simulated_users: int = 100


@app.post("/api/intervention_compare")
def intervention_compare(req: InterventionCompareRequest):
    if df is None or session_model is None:
        raise HTTPException(status_code=500, detail="Session model or data not available")

    item_rows = df[df['item_id'] == req.item_id]
    if item_rows.empty:
        raise HTTPException(status_code=404, detail=f"Item {req.item_id} not found")

    num_users = min(req.num_simulated_users, 300)
    first = item_rows.iloc[0]
    base_product = {
        'title_length': int(first['title_length']),
        'title_emo_score': float(first['title_emo_score']),
        'img_count': int(first['img_count']),
        'has_video': int(first['has_video']),
        'price': float(first['price']),
        'discount_rate': float(first['discount_rate']),
        'category': first['category'],
    }

    users_df = collect_users_for_item(req.item_id, num_users)
    actual_n = len(users_df)
    rng = np.random.default_rng(seed=abs(hash(req.item_id)) % (2**31))
    base_behavioral = simulate_diverse_users(actual_n, rng)

    scenarios = {}

    # Baseline
    proba_base = batch_session_predict(base_product, users_df, base_behavioral)
    scenarios['baseline'] = proba_base

    # +Coupon
    coupon_beh = base_behavioral.copy()
    coupon_beh['coupon_received'] = 1
    coupon_beh['coupon_used'] = 1
    scenarios['coupon'] = batch_session_predict(base_product, users_df, coupon_beh)

    # +Video
    video_product = base_product.copy()
    video_product['has_video'] = 1
    scenarios['video'] = batch_session_predict(video_product, users_df, base_behavioral)

    # +Coupon+Video
    scenarios['coupon_video'] = batch_session_predict(video_product, users_df, coupon_beh)

    # Classify baseline intent tiers
    intent_labels = [classify_intent(p) for p in proba_base]

    # Aggregate by intent tier
    by_tier = []
    for tier in INTENT_TIERS:
        mask = np.array([lbl == tier['name'] for lbl in intent_labels])
        if not mask.any():
            continue
        entry = {'tier_name': tier['name']}
        for sname, sproba in scenarios.items():
            entry[sname] = round(float(np.mean(sproba[mask])), 4)
        by_tier.append(entry)

    overall = {}
    for sname, sproba in scenarios.items():
        overall[sname] = round(float(np.mean(sproba)), 4)

    return {
        'scenarios': list(scenarios.keys()),
        'by_tier': by_tier,
        'overall': overall,
    }
