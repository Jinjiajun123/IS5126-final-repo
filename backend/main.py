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

model = artifacts['model'] if artifacts else None
feature_cols = artifacts['feature_cols'] if artifacts else []
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
    return float(mdl.predict_proba(X)[0, 1])

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
        "model_name": artifacts.get('model_name') if artifacts else "N/A"
    }

@app.post("/api/analyze")
def analyze_product(req: AnalyzeRequest):
    if not artifacts:
        raise HTTPException(status_code=500, detail="Models not loaded")

    title_length = len(req.title.strip()) if req.title.strip() else 25
    raw_emo, cal_emo = compute_sentiment(req.title.strip())

    product_params = {
        'title_length': title_length,
        'title_emo_score': cal_emo,
        'img_count': req.img_count,
        'has_video': 0,  # Legacy parameter forced to 0
        'price': req.price,
        'discount_rate': req.discount_rate,
        'category': req.category,
    }
    
    cat_bench = benchmarks.get(req.category, {})
    if not cat_bench:
        # fallback
        cat_bench = {f: 0 for f in SELLER_CONTROLLABLE}
        cat_bench['price'] = 100
        cat_bench['discount_rate'] = 0.1
        cat_bench['img_count'] = 4
        cat_bench['title_length'] = 25
        cat_bench['title_emo_score'] = 0.6

    prob, per_cluster = predict_cluster_weighted(product_params, feature_cols, category_mapping, req.coupon)

    user_params = get_default_user_params()
    user_params['coupon_received'] = req.coupon
    feature_vec = build_feature_vector(product_params, user_params, feature_cols, category_mapping)

    domain_total, domain_breakdown = compute_domain_score(product_params, cat_bench, req.title)
    hybrid, ml_score = compute_hybrid_score(prob, domain_total)

    # Counterfactuals
    cf_results = []
    for feat in SELLER_CONTROLLABLE:
        if feat not in cat_bench: continue
        current_val = product_params.get(feat, 0)
        ideal_val = cat_bench[feat]
        _, _, delta = simulate_counterfactual(
            model, pd.Series(feature_vec, index=feature_cols),
            feat, ideal_val, feature_cols
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

def call_doubao_api(prompt_text, images_base64):
    api_key = os.environ.get("ARK_API_KEY", "c729505f-a636-472e-89e8-2a6093ba937a")
    if not api_key:
        return None

    url = "https://ark.cn-beijing.volces.com/api/v3/responses"
    content = []
    
    # Cap to max 3 images to spare payload length
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
    # This mixes the traditional ML scoring with the new mocked LLM scoring logic
    
    # 1. Base ML Logic
    raw_emo, cal_emo = compute_sentiment(req.title.strip())
    product_params = {
        'title_length': len(req.title.strip()),
        'title_emo_score': cal_emo,
        'img_count': req.img_count,
        'has_video': 1, # Defaulted to 1 as video upload is unsupported
        'price': req.price,
        'discount_rate': req.discount_rate,
        'category': req.category,
    }
    
    if req.category in benchmarks:
        cat_bench = benchmarks[req.category]
    else:
        cat_bench = {f: 0 for f in SELLER_CONTROLLABLE}
        cat_bench['price'] = 100
        cat_bench['discount_rate'] = 0.1
        cat_bench['img_count'] = 4
        cat_bench['title_length'] = 25
        cat_bench['title_emo_score'] = 0.6
    
    prob, per_cluster = predict_cluster_weighted(product_params, feature_cols, category_mapping, req.coupon)
    
    # Domain knowledge & hybrid score
    domain_total, domain_breakdown = compute_domain_score(product_params, cat_bench, req.title)
    hybrid, ml_score_normalized = compute_hybrid_score(prob, domain_total)
    
    # Extract Best Persona from the ML Baseline
    persona_context = "No specific persona matched"
    if df is not None and 'user_cluster' in df.columns:
        cat_cluster_rates = df[df['category'] == req.category].groupby('user_cluster')['label'].mean()
        if len(cat_cluster_rates) > 0:
            best_cluster = cat_cluster_rates.idxmax()
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
    coupon_desc = f"a coupon is offered" if req.coupon else "no coupon is offered"

    prompt = f"""You are a senior e-commerce consultant specializing in Taobao/Tmall product optimization.

A merchant is listing a product titled "{req.title}" in the "{req.category}" category, priced at ¥{req.price} with a {req.discount_rate*100:.0f}% discount. Currently {img_desc}, and {coupon_desc}.

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
    if req.images:
        llm_result_str = call_doubao_api(json_prompt, req.images)
        
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
