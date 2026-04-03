<template>
  <div class="diagnostic-page">
    <div class="grid layout-grid">
      <!-- Left Column: Header + Input -->
      <div class="left-col">
        <header class="page-header">
          <h1>Analytics Hub</h1>
          <p>Instant product evaluation.</p>
        </header>

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
            <label>Price</label>
            <input type="number" class="input-field" v-model.number="form.price" step="5" />
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4 mt-2 mb-4">
          <div class="form-group">
            <label>Discount Rate</label>
            <input type="number" class="input-field" v-model.number="form.discount_rate" step="0.05" min="0" max="1" />
          </div>
          <div class="form-group">
            <div class="flex items-center justify-between" style="margin-bottom: 0.25rem;">
              <label style="margin-bottom: 0;">Coupon Amount</label>
              <label class="switch">
                <input type="checkbox" v-model="form.coupon" :true-value="1" :false-value="0">
                <span class="slider round"></span>
              </label>
            </div>
            <input type="number" class="input-field" v-model.number="form.coupon_value" :disabled="!form.coupon" placeholder="¥ 0.00" :class="{'opacity-40': !form.coupon, 'bg-gray-50': !form.coupon}" />
          </div>
        </div>

        <div class="upload-zone gap-2">
          <p class="upload-title">Hero Image</p>
          <div class="drag-drop-area" :class="{ 'has-file': hasFile }" @click="$refs.fileInput.click()">
            <input type="file" ref="fileInput" hidden @change="onFileChange" accept="image/*" multiple />
            <svg v-if="!hasFile" viewBox="0 0 24 24" fill="none" class="icon mx-auto"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <svg v-else viewBox="0 0 24 24" fill="none" class="icon mx-auto text-success"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14M22 4L12 14.01l-3-3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <p v-if="!hasFile">Drop images here (up to 10)</p>
            <p v-else class="text-success">{{ form.img_count }} image(s) processed.</p>
          </div>
        </div>

        <button class="btn-primary flex justify-center items-center gap-2 mt-auto" @click="handleAnalyze" :disabled="loading">
          <svg v-if="loading" class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg>
          {{ loading ? 'Running Dual Engine...' : 'Run Analytics' }}
        </button>
      </div>
      </div>

      <!-- Results Panel -->
      <div class="results-panel flex-col gap-6" v-if="result">

        <!-- Diagnostic Report with Unified Score -->
        <div class="card report-card">
          <div class="report-header">
            <h4>Diagnostic Report</h4>
            <span class="score-number" v-if="result.unified_score !== undefined">{{ Math.round(result.unified_score) }}</span>
          </div>
          <div class="report-content mt-4">
            <div class="report-section success" v-if="result.diagnostics.strengths.length">
              <div class="report-icon">✓</div>
              <div class="report-text">
                <h5>Conversion Drivers (Strengths)</h5>
                <ul><li v-for="(s, i) in result.diagnostics.strengths" :key="i">{{ s }}</li></ul>
              </div>
            </div>
            
            <div class="report-section warning mt-4" v-if="result.diagnostics.weaknesses.length">
              <div class="report-icon">!</div>
              <div class="report-text">
                <h5>Visual Detractors (Weaknesses)</h5>
                <ul><li v-for="(w, i) in result.diagnostics.weaknesses" :key="i">{{ w }}</li></ul>
              </div>
            </div>

            <div class="report-section info mt-4" v-if="result.persona_analysis">
              <div class="report-icon">👤</div>
              <div class="report-text">
                <h5>Target Persona Analysis</h5>
                <p class="text-sm leading-relaxed mt-2">{{ result.persona_analysis }}</p>
              </div>
            </div>
          </div>
          <div class="action-bar mt-6 flex justify-between items-center px-2">
            <span class="text-secondary"></span>
            <router-link to="/generator" class="link-btn">Use AI Generator →</router-link>
          </div>
        </div>
      </div>
      
      <div v-else class="mock-phone-wrapper">
        <div class="mock-phone">
          <!-- Phone Status Bar -->
          <div class="phone-status-bar">
            <span>9:41</span>
            <div class="status-icons">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M2 22h20V2z"/></svg>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M15.67 4H14V2h-4v2H8.33C7.6 4 7 4.6 7 5.33v15.33C7 21.4 7.6 22 8.33 22h7.33c.74 0 1.34-.6 1.34-1.33V5.33C17 4.6 16.4 4 15.67 4z"/></svg>
            </div>
          </div>
          
          <!-- Mock Taobao Content -->
          <div class="taobao-content">
            <div class="tb-image-area relative">
              <template v-if="imagePreviews.length > 0">
                <img :src="imagePreviews[currentImageIndex]" alt="Product Image" class="tb-image" />
                <div v-if="imagePreviews.length > 1" class="absolute bottom-3 right-3 bg-black/40 backdrop-blur-md text-white text-xs px-2 py-0.5 rounded-full font-medium tracking-wider">
                  {{ currentImageIndex + 1 }} / {{ imagePreviews.length }}
                </div>
                <!-- Carousel Buttons -->
                <button v-if="imagePreviews.length > 1" @click="prevImage" class="absolute left-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/80 shadow text-gray-800 flex items-center justify-center font-bold pb-1 hover:bg-white transition-colors">&lsaquo;</button>
                <button v-if="imagePreviews.length > 1" @click="nextImage" class="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/80 shadow text-gray-800 flex items-center justify-center font-bold pb-1 hover:bg-white transition-colors">&rsaquo;</button>
              </template>
              <div v-else class="tb-image-placeholder">
                <svg viewBox="0 0 24 24" fill="none"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <span>No Image Uploaded</span>
              </div>
            </div>
            
            <div class="tb-price-bar">
              <div class="tb-price-left">
                <span class="currency">¥</span>
                <span class="price">{{ form.price.toFixed(2) }}</span>
              </div>
              <div class="tb-price-right">
                <span>Est. w/ Coupon</span>
              </div>
            </div>
            
            <div class="tb-title-sec" :class="{ 'has-coupon': form.coupon }">
              <div class="tb-title-text">
                <span class="tb-tag">TMALL</span>{{ form.title || 'Enter product title...' }}
              </div>
              <div class="tb-sub-tags">
                <span class="tag">Fast Refund</span>
                <span class="tag">Free Shipping</span>
                <span class="tag">7-Day Return</span>
              </div>
            </div>
            
            <div class="tb-coupon-bar" v-if="form.coupon">
              <div class="coupon-ticket">
                <div class="c-left">¥<strong>{{ form.coupon_value || 0 }}</strong> Coupon</div>
                <div class="c-right">Get</div>
              </div>
            </div>
            
            <div class="tb-category-bar">
              <div class="cat-chip">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>
                <span>Category</span>
              </div>
              <span class="val">{{ form.category }}</span>
            </div>
          </div>
          
          <!-- Mock Bottom Bar -->
          <div class="taobao-bottom">
            <div class="bot-icon"><span>Store</span></div>
            <div class="bot-icon"><span>Chat</span></div>
            <div class="bot-icon"><span>Star</span></div>
            <div class="bot-btn cart">Add to Cart</div>
            <div class="bot-btn buy">Buy Now</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { fetchConfig } from '../api'
import axios from 'axios'

const categories = ref(['服饰鞋包', '数码家电', '食品生鲜', '美妆个护', '家居日用'])
const loading = ref(false)
const result = ref(null)
const hasFile = ref(false)
const imagePreviews = ref([])
const currentImageIndex = ref(0)

const form = ref({
  title: '高级修身秋季衬衫',
  category: '服饰鞋包',
  price: 99.0,
  discount_rate: 0.1,
  img_count: 0,
  coupon: 0,
  coupon_value: 10
})

const nextImage = () => { if (imagePreviews.value.length > 0) currentImageIndex.value = (currentImageIndex.value + 1) % imagePreviews.value.length; }
const prevImage = () => { if (imagePreviews.value.length > 0) currentImageIndex.value = (currentImageIndex.value - 1 + imagePreviews.value.length) % imagePreviews.value.length; }

onMounted(async () => {
  try {
    const config = await fetchConfig()
    if (config.categories?.length) {
      categories.value = config.categories
      form.value.category = categories.value[0]
    }
  } catch (e) {
    console.warn("Could not load config", e)
  }
})

const onFileChange = (e) => {
  if (e.target.files && e.target.files.length > 0) {
    hasFile.value = true
    form.value.img_count = e.target.files.length
    imagePreviews.value = []
    currentImageIndex.value = 0
    Array.from(e.target.files).forEach(file => {
      const reader = new FileReader()
      reader.onload = (ev) => {
        imagePreviews.value.push(ev.target.result)
      }
      reader.readAsDataURL(file)
    })
  }
}

const handleAnalyze = async () => {
  loading.value = true
  try {
    const payload = { ...form.value, images: imagePreviews.value }
    const res = await axios.post('http://localhost:8000/api/evaluate_hybrid', payload)
    result.value = res.data
  } catch (e) {
    console.error(e)
    alert("Analysis failed. See console.")
  }
  loading.value = false
}

const scoreGrade = computed(() => {
  if (!result.value || result.value.unified_score === undefined) {
    return { label: '-', color: '#999', bg: '#f5f5f5' }
  }
  const s = result.value.unified_score
  if (s >= 80) return { label: 'Excellent', color: '#059669', bg: '#ECFDF5' }
  if (s >= 60) return { label: 'Good', color: '#2563EB', bg: '#EFF6FF' }
  if (s >= 30) return { label: 'Average', color: '#D97706', bg: '#FFFBEB' }
  return { label: 'Needs Work', color: '#DC2626', bg: '#FEF2F2' }
})
</script>

<style scoped>
.page-header p { margin-top: 0.5rem; font-size: 1.05rem; }
.page-header { margin-bottom: 1.5rem; }
.layout-grid { grid-template-columns: 360px 1fr; gap: 2rem; align-items: start; }
.left-col { display: flex; flex-direction: column; gap: 1.5rem; }
.input-panel { display: flex; flex-direction: column; }
.panel-title { border-bottom: 1px solid var(--border-color); padding-bottom: 1rem; margin-bottom: 1.25rem; }
.form-group label { display: block; margin-bottom: 0.25rem; font-size: 0.85rem; font-weight: 500; color: var(--text-secondary); }

.upload-zone { margin-top: 0.5rem; flex: 1; }
.upload-title { font-size: 0.85rem; font-weight: 500; color: var(--text-secondary); margin-bottom: 0.25rem; }
.drag-drop-area {
  border: 1.5px dashed var(--border-color);
  border-radius: var(--radius-sm);
  padding: 1rem;
  min-height: 100px;
  text-align: center;
  cursor: pointer;
  background: var(--bg-color);
  transition: all 0.2s;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.drag-drop-area p { font-size: 0.85rem; margin-bottom: 0; }
.drag-drop-area:hover { border-color: var(--text-primary); }
.drag-drop-area.has-file { border-color: var(--success); border-style: solid; background: var(--success-bg); }
.icon { width: 24px; height: 24px; color: var(--text-muted); margin-bottom: 0.5rem; }
.text-success { color: var(--success); font-weight: 500; }
.mt-auto { margin-top: auto; }
.mt-4 { margin-top: 1rem; }
.mt-6 { margin-top: 1.5rem; }
.px-2 { padding: 0 0.5rem; }

/* Custom Toggle Switch */
.switch { position: relative; display: inline-block; width: 32px; height: 18px; margin-bottom: 0; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #d1d5db; transition: .3s; border-radius: 18px; }
.slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 2px; bottom: 2px; background-color: white; transition: .3s; border-radius: 50%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
input:checked + .slider { background-color: #ff9800; }
input:checked + .slider:before { transform: translateX(14px); }

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.score-number {
  font-size: 1.3rem;
  font-weight: 800;
  line-height: 1;
  font-family: 'Inter', system-ui, sans-serif;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.score-label {
  font-size: 0.55rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.15rem;
}

.report-card { padding: 2.5rem; position: relative; }
.report-section { display: flex; gap: 1.25rem; padding: 1.5rem; border-radius: var(--radius-sm); border-left: 3px solid; }
.report-section.success { background: var(--bg-color); border-left-color: var(--success); }
.report-section.warning { background: var(--bg-color); border-left-color: var(--warning); }
.report-section.info { background: #EEF2FF; border-left-color: #6366F1; }
.report-icon { align-self: flex-start; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: bold; font-size: 1rem; color: white; flex-shrink: 0; }
.success .report-icon { background: var(--success); }
.warning .report-icon { background: var(--warning); }
.info .report-icon { background: #6366F1; font-size: 0.85rem; }
.report-text h5 { margin: 0 0 0.5rem 0; font-size: 1rem; }
.report-text ul { margin: 0; padding-left: 1.25rem; font-size: 0.95rem; color: var(--text-primary); }
.report-text li { margin-bottom: 0.4rem; line-height: 1.5; }

.action-bar { border-top: 1px solid var(--border-color); padding-top: 1.5rem; }
.link-btn { font-size: 0.95rem; color: var(--text-primary); text-decoration: none; font-weight: 600; border-bottom: 1px solid transparent; transition: border-color 0.2s; }
.link-btn:hover { border-bottom-color: var(--text-primary); }

/* Mock Phone Styles */
.mock-phone-wrapper {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 0 1rem;
  margin-top: -7.5rem; /* Push up to the extreme visual center aligning with page header */
  position: sticky;
  top: 3rem; /* Stays in view while scrolling */
}

.mock-phone {
  width: 320px;
  height: 640px;
  background: #f2f2f2;
  border-radius: 36px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.1), inset 0 0 0 10px #09090B;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.phone-status-bar {
  display: flex;
  justify-content: space-between;
  padding: 0.8rem 1.5rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: #333;
  background: transparent;
  position: absolute;
  top: 0; left: 0; right: 0;
  z-index: 10;
}

.status-icons { display: flex; gap: 4px; }

.taobao-content {
  flex: 1;
  overflow-y: auto;
  background: #f2f2f2;
  padding-bottom: 60px;
}

.tb-image-area {
  width: 100%;
  height: 320px;
  background: #fff;
}
.tb-image { width: 100%; height: 100%; object-fit: cover; }
.tb-image-placeholder { 
  width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #999; 
  background: #E4E4E7;
}
.tb-image-placeholder svg { width: 40px; height: 40px; margin-bottom: 0.5rem; }

.tb-price-bar {
  background: linear-gradient(90deg, #ff5000, #ff8c00);
  color: #fff;
  padding: 0.8rem 1rem;
  display: flex; justify-content: space-between; align-items: flex-end;
}
.currency { font-size: 1rem; font-weight: bold; margin-right: 2px; }
.price { font-size: 1.8rem; font-weight: 800; line-height: 1; }
.tb-price-right { font-size: 0.75rem; background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 10px; }

.tb-title-sec {
  background: #fff;
  padding: 1rem;
  margin-bottom: 0.5rem;
  border-radius: 0 0 12px 12px;
}
.tb-title-sec.has-coupon { margin-bottom: 0; border-radius: 0; padding-bottom: 0.5rem; }
.tb-title-text { font-size: 1.05rem; font-weight: 600; color: #111; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.tb-tag { background: #ff0036; color: #fff; font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; margin-right: 6px; vertical-align: middle; position: relative; top: -1px; }

.tb-sub-tags { display: flex; gap: 8px; margin-top: 0.8rem; }
.tb-sub-tags .tag { font-size: 0.65rem; color: #ff5000; background: rgba(255,80,0,0.08); padding: 2px 6px; border-radius: 4px; }

.tb-coupon-bar { background: #fff; padding: 0 1rem 0.8rem; border-radius: 0 0 12px 12px; margin-bottom: 0.5rem; display: flex; }
.coupon-ticket { background: #fff0f0; border: 1px solid #ffb4b4; border-radius: 4px; display: flex; align-items: center; }
.coupon-ticket .c-left { color: #ff0036; padding: 2px 8px; font-size: 0.75rem; border-right: 1px dashed #ffb4b4; display: flex; align-items: baseline; }
.coupon-ticket .c-left strong { font-size: 1.1rem; font-weight: 800; margin: 0 2px; }
.coupon-ticket .c-right { color: #ff0036; padding: 2px 8px; font-size: 0.75rem; font-weight: 700; background: #ffe4e4; }

.tb-category-bar {
  background: var(--accent-soft);
  padding: 1.25rem 1rem;
  border-radius: 16px;
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.95rem;
  margin-top: 1rem;
  border: 1px solid rgba(255,80,0,0.1);
}
.tb-category-bar .cat-chip { display: flex; align-items: center; gap: 0.5rem; color: var(--accent-color); font-weight: 600; }
.tb-category-bar .val { color: #1E293B; font-weight: 700; background: #fff; padding: 0.4rem 0.8rem; border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }

.taobao-bottom {
  position: absolute; bottom: 0; left: 0; right: 0;
  height: 65px; background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); border-top: 1px solid #eee;
  display: flex; align-items: center; padding: 0 0.5rem;
}
.bot-icon { flex: 1; display: flex; flex-direction: column; align-items: center; font-size: 0.6rem; color: #666; font-weight: 500; }
.bot-icon span { margin-top: 22px; } /* Simple layout hack for icons */
.bot-icon::before { content: ""; position: absolute; width: 20px; height: 20px; border: 1.5px solid #666; border-radius: 50%; top: 10px; }
.bot-btn { width: 95px; height: 42px; border-radius: 21px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.85rem; font-weight: 700; margin-left: 0.5rem; box-shadow: 0 4px 10px rgba(255,80,0,0.3); }
.bot-btn.cart { background: linear-gradient(90deg, #ffcb00, #ff9402); color: #fff; }
.bot-btn.buy { background: linear-gradient(90deg, #ff5000, #ff8c00); }

</style>
