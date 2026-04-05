<template>
  <div class="audience-page">
    <div class="grid layout-grid">
      <!-- Left Column: Product Selection -->
      <div class="left-col">
        <header class="page-header">
          <h1>User Insights</h1>
          <p>Session-based user intent analysis &amp; actionable recommendations.</p>
        </header>

        <div class="card input-panel">
          <h3 class="panel-title">Select Product</h3>

          <div class="form-group">
            <label>Category Filter</label>
            <select class="input-field" v-model="selectedCategory" @change="onCategoryChange">
              <option value="">All Categories</option>
              <option v-for="cat in categories" :key="cat" :value="cat">{{ catEN(cat) }}</option>
            </select>
          </div>

          <div class="form-group">
            <label>Product</label>
            <select class="input-field" v-model="selectedItemId" @change="onProductChange">
              <option value="">-- Select a product --</option>
              <option v-for="p in products" :key="p.item_id" :value="p.item_id">
                {{ p.item_id }} | {{ catEN(p.category) }} | &#165;{{ p.price }}
              </option>
            </select>
          </div>

          <!-- Product Summary -->
          <div v-if="selectedProduct" class="product-summary">
            <div class="summary-row"><span class="label">Category</span><span class="value">{{ catEN(selectedProduct.category) }}</span></div>
            <div class="summary-row"><span class="label">Price</span><span class="value">&#165;{{ selectedProduct.price.toFixed(2) }}</span></div>
            <div class="summary-row"><span class="label">Discount</span><span class="value">{{ (selectedProduct.discount_rate * 100).toFixed(0) }}%</span></div>
            <div class="summary-row"><span class="label">Images</span><span class="value">{{ selectedProduct.img_count }}</span></div>
            <div class="summary-row"><span class="label">Interactions</span><span class="value">{{ selectedProduct.interaction_count }}</span></div>
            <div class="summary-divider"></div>
            <div class="summary-row"><span class="label">Likes</span><span class="value">{{ selectedProduct.like_num }}</span></div>
            <div class="summary-row"><span class="label">Collects</span><span class="value">{{ selectedProduct.collect_num }}</span></div>
            <div class="summary-row"><span class="label">Comments</span><span class="value">{{ selectedProduct.comment_num }}</span></div>
            <div class="summary-row"><span class="label">Shares</span><span class="value">{{ selectedProduct.share_num }}</span></div>
            <div class="summary-divider"></div>
            <div class="summary-row"><span class="label">Purchase Rate</span><span class="value highlight">{{ (selectedProduct.purchase_rate * 100).toFixed(1) }}%</span></div>
          </div>

          <button class="btn-primary flex justify-center items-center gap-2 mt-auto" @click="handleAnalyze" :disabled="loading || !selectedItemId">
            <svg v-if="loading" class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg>
            {{ loading ? 'Analyzing...' : 'Analyze Users' }}
          </button>
        </div>
      </div>

      <!-- Right Column: Results -->
      <div class="results-panel" v-if="result">

        <!-- Overall Stats Bar -->
        <div class="card stats-bar">
          <div class="stat-item">
            <span class="stat-value">{{ result.overall_stats.total_simulated }}</span>
            <span class="stat-label">Users Analyzed</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ (result.overall_stats.mean_prob * 100).toFixed(1) }}%</span>
            <span class="stat-label">Avg. Purchase Prob</span>
          </div>
          <div class="stat-item accent">
            <span class="stat-value">{{ result.overall_stats.high_intent_count }}</span>
            <span class="stat-label">High-Intent Users</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ (result.overall_stats.high_intent_pct * 100).toFixed(1) }}%</span>
            <span class="stat-label">High-Intent Rate</span>
          </div>
        </div>

        <!-- Single Chart: Probability Distribution with tier annotation -->
        <div class="card">
          <h4>Purchase Probability Distribution</h4>
          <p class="text-sm text-secondary mb-4">User count by predicted purchase probability. Colors indicate intent tier: <span class="tier-label-hi">High</span> / <span class="tier-label-med">Medium</span> / <span class="tier-label-lo">Low</span></p>
          <v-chart :option="histOption" style="height: 300px;" autoresize />
        </div>

      </div>

      <!-- ═══ FULL-WIDTH RECOMMENDATIONS SECTION (below the two-column grid) ═══ -->
      <div class="full-width-section" v-if="result">

        <div class="section-header">
          <h3>Actionable Recommendations</h3>
          <p>Data-driven strategies to maximize conversion across all user segments.</p>
        </div>

        <!-- Rec 1 & 2: Coupon Targeting + High-Intent Decision (side by side) -->
        <div class="grid grid-cols-2 gap-6">
          <div class="card rec-card coupon-card" v-if="result.recommendations.coupon_targeting">
            <div class="rec-badge coupon-badge">Best ROI</div>
            <h4>{{ result.recommendations.coupon_targeting.title }}</h4>
            <p class="rec-big-num">{{ result.recommendations.coupon_targeting.target_count }}<span class="rec-unit"> users</span></p>
            <p class="rec-desc">{{ result.recommendations.coupon_targeting.description }}</p>
            <div class="rec-lift" v-if="result.recommendations.coupon_targeting.estimated_lift">
              Estimated lift: <strong>+{{ (result.recommendations.coupon_targeting.estimated_lift * 100).toFixed(1) }}pp</strong>
            </div>
            <p class="rec-action">{{ result.recommendations.coupon_targeting.action }}</p>
            <button class="btn-action btn-coupon" @click="handleSendCoupons">Send Coupons</button>
          </div>

          <div class="card rec-card hi-coupon-card" v-if="result.recommendations.high_intent_coupon">
            <div :class="['rec-badge', result.recommendations.high_intent_coupon.verdict === 'no_coupon' ? 'no-coupon-badge' : 'send-coupon-badge']">
              {{ result.recommendations.high_intent_coupon.verdict === 'no_coupon' ? 'Save Margin' : result.recommendations.high_intent_coupon.verdict === 'send_coupon' ? 'Send Coupon' : 'N/A' }}
            </div>
            <h4>{{ result.recommendations.high_intent_coupon.title }}</h4>
            <p class="rec-big-num">{{ result.recommendations.high_intent_coupon.user_count }}<span class="rec-unit"> users</span></p>
            <p class="rec-desc">{{ result.recommendations.high_intent_coupon.description }}</p>
            <div class="rec-lift">
              Coupon lift: <strong>{{ result.recommendations.high_intent_coupon.coupon_lift >= 0 ? '+' : '' }}{{ (result.recommendations.high_intent_coupon.coupon_lift * 100).toFixed(1) }}pp</strong>
            </div>
          </div>
        </div>

        <!-- Rec 3 & 4: Traffic Push + Low-Intent Uplift (side by side) -->
        <div class="grid grid-cols-2 gap-6">
          <div class="card rec-card traffic-card" v-if="result.recommendations.traffic_push">
            <div class="rec-badge traffic-badge">Traffic</div>
            <h4>{{ result.recommendations.traffic_push.title }}</h4>
            <p class="rec-desc">{{ result.recommendations.traffic_push.description }}</p>
            <div class="traffic-stats-row">
              <div class="traffic-stat">
                <span class="stat-value">{{ result.recommendations.traffic_push.target_count }}</span>
                <span class="stat-label">Candidates</span>
              </div>
              <div class="traffic-stat">
                <span class="stat-value">{{ (result.recommendations.traffic_push.avg_prob * 100).toFixed(1) }}%</span>
                <span class="stat-label">Avg Prob</span>
              </div>
            </div>
            <p class="rec-action">{{ result.recommendations.traffic_push.action }}</p>
            <button class="btn-action btn-traffic" @click="handleStartTraffic">Start Traffic Push</button>
          </div>

          <div class="card rec-highlight low-highlight" v-if="result.recommendations.low_intent_uplift">
            <div class="rec-highlight-header">
              <div class="rec-badge low-badge">Priority Action</div>
              <h4>{{ result.recommendations.low_intent_uplift.title }}</h4>
              <p class="rec-meta">{{ result.recommendations.low_intent_uplift.user_count }} users ({{ (result.recommendations.low_intent_uplift.user_pct * 100).toFixed(1) }}%) in this group</p>
            </div>
            <ul class="insight-list" v-if="result.recommendations.low_intent_uplift.insights.length">
              <li v-for="(insight, i) in result.recommendations.low_intent_uplift.insights" :key="i">{{ insight }}</li>
            </ul>
            <p v-else class="text-muted">No low-intent users detected. Your product performs well across the board.</p>
          </div>
        </div>

        <!-- High Intent Users Table -->
        <div class="card">
          <h4>Top High-Intent Users</h4>
          <p class="text-sm text-secondary mb-4">Users most likely to convert, ranked by predicted purchase probability. <span class="text-muted">(Simulated session data)</span></p>
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>User ID</th>
                  <th>Intent Tier</th>
                  <th>Purchase Prob</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th>Spend</th>
                  <th>Cart</th>
                  <th>PV</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(u, idx) in result.high_intent_users" :key="u.user_id">
                  <td>{{ idx + 1 }}</td>
                  <td class="mono">{{ u.user_id }}</td>
                  <td><span :class="['tier-badge', tierClass(u.intent_tier)]">{{ u.intent_tier }}</span></td>
                  <td>
                    <div class="prob-bar-cell">
                      <div class="prob-bar" :style="{ width: (u.prob * 100) + '%', background: probColor(u.prob) }"></div>
                      <span class="prob-text">{{ (u.prob * 100).toFixed(1) }}%</span>
                    </div>
                  </td>
                  <td>{{ u.age }}</td>
                  <td>{{ u.gender }}</td>
                  <td>&#165;{{ u.total_spend.toFixed(0) }}</td>
                  <td>{{ u.add2cart ? 'Yes' : '-' }}</td>
                  <td>{{ u.pv_count }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>

      <!-- Empty state -->
      <div v-else class="empty-state">
        <div class="empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
            <circle cx="9" cy="7" r="4"></circle>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
          </svg>
        </div>
        <h3>Select a product to analyze</h3>
        <p>Choose a listed product from the left panel, then click "Analyze Users" to identify user intent segments and get actionable recommendations.</p>
      </div>
    </div>

    <!-- Toast -->
    <transition name="toast">
      <div v-if="showToast" class="toast">{{ toastMsg }}</div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { fetchProducts, analyzeSession, fetchConfig } from '../api'

const CATEGORY_EN = {
  '服饰鞋包': 'Clothing, Shoes & Bags',
  '数码家电': 'Electronics & Appliances',
  '食品生鲜': 'Food & Groceries',
  '美妆个护': 'Beauty & Personal Care',
  '家居日用': 'Home & Living',
  '其他': 'Others',
}
const catEN = (name) => CATEGORY_EN[name] || name

const categories = ref([])
const products = ref([])
const selectedCategory = ref('')
const selectedItemId = ref('')
const loading = ref(false)
const result = ref(null)
const showToast = ref(false)
const toastMsg = ref('')

const selectedProduct = computed(() => {
  if (!selectedItemId.value) return null
  return products.value.find(p => p.item_id === selectedItemId.value) || null
})

const loadProducts = async () => {
  try {
    const params = { limit: 100 }
    if (selectedCategory.value) params.category = selectedCategory.value
    const data = await fetchProducts(params)
    products.value = data.items || []
  } catch (e) {
    console.error('Failed to load products:', e)
  }
}

onMounted(async () => {
  try {
    const config = await fetchConfig()
    categories.value = config.categories || []
  } catch (e) {
    console.error('Failed to load config:', e)
  }
  await loadProducts()
})

const onCategoryChange = () => {
  selectedItemId.value = ''
  result.value = null
  loadProducts()
}

const onProductChange = () => {
  result.value = null
}

const handleAnalyze = async () => {
  if (!selectedItemId.value) return
  loading.value = true
  result.value = null
  try {
    result.value = await analyzeSession({ item_id: selectedItemId.value, num_simulated_users: 200 })
  } catch (e) {
    console.error('Analysis failed:', e)
    toast('Analysis failed. Check console.')
  }
  loading.value = false
}

const handleSendCoupons = () => {
  toast('Coupons sent to targeted medium-intent users!')
}

const handleStartTraffic = () => {
  toast('Traffic push campaign started for high-intent candidates!')
}

const toast = (msg) => {
  toastMsg.value = msg
  showToast.value = true
  setTimeout(() => { showToast.value = false }, 3000)
}

const probColor = (p) => {
  if (p >= 0.55) return '#10B981'
  if (p >= 0.35) return '#F59E0B'
  return '#EF4444'
}

const tierClass = (tier) => {
  if (tier === 'High Intent') return 'tier-high'
  if (tier === 'Medium Intent') return 'tier-medium'
  return 'tier-low'
}

// ECharts: single histogram with tier coloring + markArea
const histOption = computed(() => {
  if (!result.value) return {}
  const bins = result.value.prob_distribution.bins
  const counts = result.value.prob_distribution.counts
  const labels = []
  const colors = []
  for (let i = 0; i < counts.length; i++) {
    labels.push(`${(bins[i] * 100).toFixed(0)}-${(bins[i + 1] * 100).toFixed(0)}%`)
    const mid = (bins[i] + bins[i + 1]) / 2
    if (mid >= 0.55) colors.push('#10B981')
    else if (mid >= 0.35) colors.push('#F59E0B')
    else colors.push('#EF4444')
  }
  // Tier summary for subtitle
  const tiers = result.value.tier_distribution || []
  const tierSummary = tiers.map(t => `${t.tier_name}: ${t.user_count} (${(t.user_pct * 100).toFixed(0)}%)`).join('  |  ')

  return {
    tooltip: { trigger: 'axis', formatter: (params) => {
      const p = params[0]
      return `${p.name}<br/>Users: <b>${p.value}</b>`
    }},
    xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', name: 'Users' },
    series: [{
      type: 'bar',
      data: counts.map((c, i) => ({ value: c, itemStyle: { color: colors[i], borderRadius: [4, 4, 0, 0] } })),
      barWidth: '65%',
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { type: 'dashed', color: '#9CA3AF', width: 1 },
        label: { fontSize: 10, color: '#6B7280' },
        data: [
          { xAxis: '30-40%', label: { formatter: 'Med', position: 'start' } },
          { xAxis: '50-60%', label: { formatter: 'High', position: 'start' } },
        ]
      }
    }],
    grid: { left: 50, right: 20, top: 20, bottom: 55 },
    graphic: [{
      type: 'text',
      left: 'center',
      bottom: 0,
      style: { text: tierSummary, fontSize: 11, fill: '#6B7280' }
    }]
  }
})
</script>

<style scoped>
.page-header p { margin-top: 0.5rem; font-size: 1.05rem; }
.page-header { margin-bottom: 1.5rem; }
.layout-grid { grid-template-columns: 340px 1fr; gap: 2rem; align-items: start; }
.left-col { display: flex; flex-direction: column; gap: 1.5rem; }
.input-panel { display: flex; flex-direction: column; }
.panel-title { border-bottom: 1px solid var(--border-color); padding-bottom: 1rem; margin-bottom: 1.25rem; }
.form-group label { display: block; margin-bottom: 0.25rem; font-size: 0.85rem; font-weight: 500; color: var(--text-secondary); }
.mt-auto { margin-top: auto; }

.product-summary {
  background: var(--bg-color);
  border-radius: var(--radius-sm);
  padding: 1rem;
  margin: 0.75rem 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.summary-row { display: flex; justify-content: space-between; font-size: 0.85rem; }
.summary-row .label { color: var(--text-secondary); }
.summary-row .value { font-weight: 600; color: var(--text-primary); }
.summary-row .value.highlight { color: var(--accent-color); }
.summary-divider { height: 1px; background: var(--border-color); margin: 0.25rem 0; }

.results-panel { display: flex; flex-direction: column; gap: 1.5rem; }

/* Stats Bar */
.stats-bar { display: flex; gap: 1rem; justify-content: space-around; padding: 1.5rem; }
.stat-item { text-align: center; }
.stat-value { display: block; font-size: 1.6rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }
.stat-label { font-size: 0.75rem; color: var(--text-secondary); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
.stat-item.accent .stat-value { color: var(--accent-color); }

/* Grid helpers */
.grid-cols-2 { grid-template-columns: 1fr 1fr; }
.gap-6 { gap: 1rem; }
.mb-4 { margin-bottom: 1rem; }
.text-sm { font-size: 0.85rem; }
.text-secondary { color: var(--text-secondary); }
.text-muted { color: var(--text-muted); }
.mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; }

/* Tier inline labels in chart description */
.tier-label-hi { color: #10B981; font-weight: 600; }
.tier-label-med { color: #F59E0B; font-weight: 600; }
.tier-label-lo { color: #EF4444; font-weight: 600; }

/* Full-width recommendations section */
.full-width-section { grid-column: 1 / -1; display: flex; flex-direction: column; gap: 1rem; margin-top: 0.25rem; }

/* Section header */
.section-header { margin-top: 0.25rem; }
.section-header h3 { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.15rem; }
.section-header p { font-size: 0.82rem; color: var(--text-secondary); }

/* ═══ Recommendation Cards ═══ */
.rec-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.25rem;
}
.low-badge { background: #FEE2E2; color: #DC2626; }
.coupon-badge { background: #FFF7ED; color: #EA580C; }
.no-coupon-badge { background: #D1FAE5; color: #059669; }
.send-coupon-badge { background: #FEF3C7; color: #D97706; }
.traffic-badge { background: #DBEAFE; color: #2563EB; }

/* Low-Intent Highlight Card */
.rec-highlight {
  padding: 1rem;
  border-left: 4px solid;
}
.low-highlight {
  border-left-color: #EF4444;
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.03), transparent);
}
.rec-highlight-header h4 {
  font-size: 0.95rem;
  margin-bottom: 0.15rem;
}
.rec-meta {
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
}
.insight-list {
  margin: 0;
  padding-left: 1.25rem;
}
.insight-list li {
  font-size: 0.8rem;
  line-height: 1.5;
  margin-bottom: 0.2rem;
  color: var(--text-primary);
}

/* Mid-level rec cards */
.rec-card {
  padding: 1rem;
}
.coupon-card { border-top: 3px solid #EA580C; }
.hi-coupon-card { border-top: 3px solid #10B981; }
.traffic-card { border-top: 3px solid #2563EB; }
.rec-big-num {
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--text-primary);
  margin: 0.15rem 0;
}
.rec-unit { font-size: 0.8rem; font-weight: 500; color: var(--text-secondary); }
.rec-desc {
  font-size: 0.8rem;
  color: var(--text-secondary);
  line-height: 1.45;
  margin: 0.3rem 0;
}
.rec-lift {
  font-size: 0.78rem;
  color: var(--text-secondary);
  margin-top: 0.3rem;
  padding: 0.25rem 0.6rem;
  background: var(--bg-color);
  border-radius: var(--radius-sm);
  display: inline-block;
}
.rec-lift strong { color: var(--accent-color); }
.rec-action {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--accent-color);
  margin-top: 0.4rem;
}

/* Action buttons */
.btn-action {
  display: inline-block;
  margin-top: 0.5rem;
  padding: 0.4rem 1.2rem;
  border: none;
  border-radius: var(--radius-sm);
  color: #fff;
  font-weight: 700;
  font-size: 0.8rem;
  cursor: pointer;
  transition: transform 0.15s, box-shadow 0.15s;
}
.btn-action:hover { transform: translateY(-1px); }
.btn-coupon {
  background: linear-gradient(135deg, #FF5000, #EA580C);
  box-shadow: 0 2px 8px rgba(255, 80, 0, 0.25);
}
.btn-coupon:hover { box-shadow: 0 4px 14px rgba(255, 80, 0, 0.35); }
.btn-traffic {
  background: linear-gradient(135deg, #2563EB, #1D4ED8);
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
}
.btn-traffic:hover { box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35); }

/* Traffic card layout */
.traffic-stats-row {
  display: flex;
  gap: 1.5rem;
  margin: 0.75rem 0;
}
.traffic-stat { text-align: center; }
.traffic-stat .stat-value { font-size: 1.4rem; }
.traffic-stat .stat-label { font-size: 0.7rem; }

/* Table */
.table-wrapper { overflow-x: auto; max-height: 400px; overflow-y: auto; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.data-table th { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 2px solid var(--border-color); font-weight: 600; color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; position: sticky; top: 0; background: var(--surface-color); z-index: 1; }
.data-table td { padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border-color); }
.data-table tbody tr:hover { background: var(--bg-color); }

/* Prob bar */
.prob-bar-cell { position: relative; width: 100%; min-width: 100px; }
.prob-bar { height: 20px; border-radius: 4px; transition: width 0.3s; }
.prob-text { position: absolute; top: 1px; left: 6px; font-size: 0.75rem; font-weight: 600; color: #fff; line-height: 20px; text-shadow: 0 1px 2px rgba(0,0,0,0.2); }

/* Tier badge */
.tier-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  white-space: nowrap;
}
.tier-high { background: #D1FAE5; color: #059669; }
.tier-medium { background: #FEF3C7; color: #D97706; }
.tier-low { background: #FEE2E2; color: #DC2626; }

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  text-align: center;
  color: var(--text-muted);
}
.empty-icon svg { width: 64px; height: 64px; margin-bottom: 1.5rem; opacity: 0.3; }
.empty-state h3 { color: var(--text-secondary); margin-bottom: 0.5rem; }
.empty-state p { max-width: 360px; font-size: 0.9rem; line-height: 1.6; }

/* Toast */
.toast {
  position: fixed;
  bottom: 2rem;
  left: 50%;
  transform: translateX(-50%);
  background: #1C1C1E;
  color: #fff;
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-sm);
  font-size: 0.9rem;
  font-weight: 500;
  box-shadow: 0 8px 24px rgba(0,0,0,0.2);
  z-index: 1000;
}
.toast-enter-active { transition: all 0.3s ease; }
.toast-leave-active { transition: all 0.3s ease; }
.toast-enter-from { opacity: 0; transform: translateX(-50%) translateY(20px); }
.toast-leave-to { opacity: 0; transform: translateX(-50%) translateY(20px); }

/* Spinner */
.spinner { animation: rotate 1s linear infinite; width: 20px; height: 20px; }
.spinner .path { stroke: white; stroke-linecap: round; animation: dash 1.5s ease-in-out infinite; }
@keyframes rotate { 100% { transform: rotate(360deg); } }
@keyframes dash {
  0% { stroke-dasharray: 1, 150; stroke-dashoffset: 0; }
  50% { stroke-dasharray: 90, 150; stroke-dashoffset: -35; }
  100% { stroke-dasharray: 90, 150; stroke-dashoffset: -124; }
}
</style>
