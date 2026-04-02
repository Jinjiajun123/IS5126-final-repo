<template>
  <div class="analyzer-page">
    <header class="page-header">
      <h1>Product Analyzer</h1>
      <p>Evaluate product parameters, discover optimization points, and view performance predictions.</p>
    </header>

    <div class="grid layout-grid">
      <!-- Input Panel -->
      <div class="card input-panel">
        <h3 class="panel-title">Product Details</h3>
        
        <div class="form-group">
          <label>Product Title</label>
          <input type="text" class="input-field" v-model="form.title" placeholder="e.g., 高级修身秋季衬衫" />
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="form-group">
            <label>Category</label>
            <select class="input-field" v-model="form.category">
              <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
            </select>
          </div>
          <div class="form-group">
            <label>Image Count</label>
            <input type="number" class="input-field" v-model.number="form.img_count" min="0" max="15" />
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="form-group">
            <label>Price</label>
            <input type="number" class="input-field" v-model.number="form.price" step="5" />
          </div>
          <div class="form-group">
            <label>Discount Rate</label>
            <input type="number" class="input-field" v-model.number="form.discount_rate" step="0.01" />
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="form-group">
            <label>Has Video?</label>
            <select class="input-field" v-model.number="form.has_video">
              <option :value="1">Yes</option>
              <option :value="0">No</option>
            </select>
          </div>
          <div class="form-group">
            <label>Offer Coupon?</label>
            <select class="input-field" v-model.number="form.coupon">
              <option :value="1">Yes</option>
              <option :value="0">No</option>
            </select>
          </div>
        </div>

        <button class="btn-primary w-full mt-4" @click="handleAnalyze" :disabled="loading">
          {{ loading ? 'Analyzing...' : 'Analyze Product' }}
        </button>
      </div>

      <!-- Results Panel -->
      <div class="results-panel flex-col gap-6" v-if="result">
        
        <!-- Score Overview -->
        <div class="grid grid-cols-2 gap-6">
          <div class="card score-card">
            <h4>Purchase Probability</h4>
            <div class="chart-wrapper">
              <v-chart class="chart" :option="gaugeOption(result.prob * 100, 'Probability', '#0A84FF')" autoresize />
            </div>
            <p v-if="result.pop_est" class="subtle-stats">
              Est. Likes: <b>{{ result.pop_est.like_est.toFixed(0) }}</b> | Est. Collections: <b>{{ result.pop_est.collect_est.toFixed(0) }}</b>
            </p>
          </div>
          
          <div class="card score-card">
            <h4>Domain Knowledge Score</h4>
            <div class="chart-wrapper">
              <v-chart class="chart" :option="gaugeOption(result.domain_total, 'Score', '#34C759')" autoresize />
            </div>
            <p class="subtle-stats">Hybrid Score: <b>{{ result.hybrid_score.toFixed(1) }}</b> / 100</p>
          </div>
        </div>

        <!-- Target Persona & Insights -->
        <div class="grid grid-cols-2 gap-6">
          
          <div class="card persona-card" v-if="result.persona">
            <h4>Target User Segment</h4>
            <h2 class="persona-title">{{ result.persona.cluster_name }}</h2>
            <div class="persona-stats">
              <div class="stat"><label>Purchase Rate</label><span>{{ (result.persona.purchase_rate * 100).toFixed(1) }}%</span></div>
              <div class="stat"><label>Avg Age</label><span>{{ result.persona.avg_age.toFixed(0) }}</span></div>
              <div class="stat"><label>Avg Spend</label><span>¥{{ result.persona.avg_spend.toLocaleString() }}</span></div>
              <div class="stat"><label>Gender</label><span>{{ result.persona.gender }}</span></div>
            </div>
          </div>
          
          <div class="card recom-card">
            <h4>Optimization Opportunities</h4>
            <div class="recom-list">
              <div class="recom-item" v-for="cf in sortedCfs" :key="cf.feature">
                <div class="cf-info">
                  <span class="cf-name">{{ featureLabels[cf.feature] || cf.feature }}</span>
                  <span class="cf-current">Current: {{ cf.current.toFixed(1) }} → Target: {{ cf.benchmark.toFixed(1) }}</span>
                </div>
                <div :class="['cf-delta', cf.delta > 0 ? 'text-success' : 'text-danger']">
                  {{ cf.delta > 0 ? '+' : '' }}{{ (cf.delta * 100).toFixed(1) }}% lift
                </div>
              </div>
            </div>
          </div>
          
        </div>
      </div>
      <div v-else class="empty-state">
        <p>No analysis requested yet. Enter product parameters and click Analyze.</p>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { fetchConfig, analyzeProduct } from '../api'

const categories = ref([])
const loading = ref(false)
const result = ref(null)

const form = ref({
  title: '夏季新款时尚女装连衣裙 轻薄透气修身显瘦百搭',
  category: '服饰鞋包',
  img_count: 5,
  price: 99.0,
  discount_rate: 0.1,
  has_video: 1,
  coupon: 0
})

const featureLabels = {
  'title_length': 'Title Length',
  'title_emo_score': 'Title Sentiment',
  'img_count': 'Image Count',
  'has_video': 'Has Video',
  'price': 'Price',
  'discount_rate': 'Discount'
}

onMounted(async () => {
  try {
    const config = await fetchConfig()
    categories.value = config.categories
    if (categories.value.length > 0 && !categories.value.includes(form.value.category)) {
      form.value.category = categories.value[0]
    }
  } catch (e) {
    console.warn("Could not load config", e)
  }
})

const handleAnalyze = async () => {
  loading.value = true
  try {
    result.value = await analyzeProduct(form.value)
  } catch (e) {
    console.error(e)
    alert("Failed to analyze. Is backend running?")
  }
  loading.value = false
}

const sortedCfs = computed(() => {
  if (!result.value || !result.value.cf_results) return []
  return [...result.value.cf_results].sort((a, b) => b.delta - a.delta).slice(0, 4) // Top 4
})

const gaugeOption = (value, name, color) => {
  return {
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        center: ['50%', '75%'],
        radius: '100%',
        min: 0,
        max: 100,
        splitNumber: 10,
        axisLine: {
          lineStyle: {
            width: 10,
            color: [
              [0.3, '#ff4500'],
              [0.7, '#ff9500'],
              [1, '#34c759']
            ]
          }
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
          length: '12%',
          width: 20,
          offsetCenter: [0, '-60%'],
          itemStyle: {
            color: 'inherit'
          }
        },
        axisTick: {
          length: 12,
          lineStyle: { color: 'inherit', width: 1 }
        },
        splitLine: {
          length: 15,
          lineStyle: { color: 'inherit', width: 2 }
        },
        axisLabel: { show: false },
        title: { show: false },
        detail: {
          fontSize: 32,
          offsetCenter: [0, '-10%'],
          valueAnimation: true,
          formatter: '{value}%',
          color: color,
          fontWeight: 600
        },
        data: [{ value: value.toFixed(1), name }]
      }
    ]
  }
}
</script>

<style scoped>
.analyzer-page {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.page-header h1 {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.layout-grid {
  grid-template-columns: 350px 1fr;
  gap: 2rem;
  align-items: start;
}

.panel-title {
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-color);
}

.form-group {
  margin-bottom: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.form-group label {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.w-full { width: 100%; }
.mt-4 { margin-top: 1rem; }

.score-card {
  text-align: center;
  padding: 2rem 1.5rem;
}

.chart-wrapper {
  height: 200px;
  margin-top: 1rem;
}

.chart {
  width: 100%;
  height: 100%;
}

.subtle-stats {
  font-size: 0.85rem;
  margin-top: 1rem;
}

.subtle-stats b {
  color: var(--text-primary);
}

.persona-title {
  color: var(--accent-color);
  margin-top: 1.5rem;
  margin-bottom: 1.5rem;
}

.persona-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.stat {
  display: flex;
  flex-direction: column;
}

.stat label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat span {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.recom-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1.5rem;
}

.recom-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: var(--bg-color);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--border-color);
}

.recom-item:hover {
  background: white;
  box-shadow: var(--shadow-sm);
}

.cf-info {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.cf-name {
  font-weight: 600;
  font-size: 0.95rem;
}

.cf-current {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.cf-delta {
  font-weight: 700;
  font-size: 1rem;
}

.text-success { color: var(--success); }
.text-danger { color: var(--danger); }

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-secondary);
  border: 1px dashed var(--border-color);
  border-radius: var(--radius-md);
  padding: 3rem;
  text-align: center;
}
</style>
