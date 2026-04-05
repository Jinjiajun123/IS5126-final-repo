<template>
  <div class="generator-page">
    <header class="page-header text-center">
      <h1>Ad Studio</h1>
      <p>Generate high-converting advertising banners and campaigns from your raw product photos.</p>
    </header>

    <div class="studio-center-layout">
      <!-- Input Panel -->
      <div class="card input-panel">
        <h3 class="panel-title mb-4">Campaign Brief</h3>
        
        <div class="form-grid">
          <div class="flex-col">
            <div class="form-group">
              <label>Item Description</label>
              <textarea class="input-field" rows="2" v-model="form.description" placeholder="A minimalist white ceramic coffee mug sitting on an oak table..."></textarea>
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div class="form-group mb-0">
                <label>Target Audience</label>
                <input type="text" class="input-field" v-model="form.audience" placeholder="e.g., 25-35 Urban" />
              </div>

              <div class="form-group mb-0">
                <label>Aesthetic Preset</label>
                <select class="input-field" v-model="form.style">
                  <option value="minimalist">Clean & Minimalist</option>
                  <option value="lifestyle">Earthy Lifestyle</option>
                  <option value="luxury">Dark High-End</option>
                  <option value="custom">Custom Style (Define Your Own)</option>
                </select>
              </div>
            </div>

            <div class="form-group mt-4 mb-0" v-if="form.style === 'custom'">
              <label>Custom Style Keywords</label>
              <input type="text" class="input-field" v-model="form.customStyle" placeholder="e.g., Cyberpunk neon lights, highly detailed, 8k..." />
            </div>
          </div>

          <div class="form-group flex-col h-full mb-0 upload-col">
            <label>Original Product Photo</label>
            <div class="upload-zone-border flex-1 flex-col justify-center" :class="{ 'has-file': hasFile }" @click="$refs.genFileInput.click()">
              <input type="file" ref="genFileInput" hidden @change="onFileChange" accept="image/*" />
              <svg v-if="!hasFile" viewBox="0 0 24 24" fill="none" class="empty-icon"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
              <span v-if="!hasFile" class="text-sm mt-2">Click to upload image</span>
              <span v-else class="text-sm success-text">Image loaded.</span>
            </div>
          </div>
        </div>

        <div class="form-group mt-6 mb-0">
          <button class="btn-primary w-full flex justify-center items-center gap-2" @click="handleGenerate" :disabled="loading">
            <svg v-if="loading" class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg>
            {{ loading ? 'Synthesizing Visuals...' : 'Generate Images' }}
          </button>
        </div>
      </div>

      <!-- Gallery -->
      <div class="gallery-container" v-if="result">
        <h3 class="panel-title text-center mt-8">Gallery</h3>
        <div class="image-grid">
          <div class="image-wrap" v-for="(img, i) in result.images" :key="i">
            <img :src="img" alt="Generative Mock" loading="lazy" />
            <div class="overlay">
              <button class="download-btn">Save Asset</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const loading = ref(false)
const result = ref(null)
const hasFile = ref(false)
const imagePreview = ref(null)

const form = ref({
  description: '一款海边度假风格的防晒衣，轻薄透气。',
  audience: '18-25岁年轻女性，大学生',
  style: 'lifestyle',
  customStyle: ''
})

const onFileChange = (e) => {
  if (e.target.files && e.target.files.length > 0) {
    hasFile.value = true
    const file = e.target.files[0]
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

const handleGenerate = async () => {
  loading.value = true
  try {
    const res = await axios.post('http://localhost:8000/api/generate_creative', {
      ...form.value,
      image: imagePreview.value || undefined
    })
    result.value = res.data
  } catch (e) {
    console.error(e)
    alert("API generation failed.")
  }
  loading.value = false
}
</script>

<style scoped>
.page-header p { margin-top: 0.5rem; font-size: 1.05rem; }
.studio-center-layout { max-width: 850px; margin: 0 auto; display: flex; flex-direction: column; gap: 2rem; }
.input-panel { display: flex; flex-direction: column; }
.panel-title { border-bottom: 1px solid var(--border-color); padding-bottom: 1.25rem; }

.form-grid { display: grid; grid-template-columns: 1fr 220px; gap: 1.5rem; }
@media (max-width: 768px) { .form-grid { grid-template-columns: 1fr; } }
.h-full { height: 100%; }
.flex-1 { flex: 1; }
.mb-0 { margin-bottom: 0 !important; }
.mb-4 { margin-bottom: 1rem; }
.mt-2 { margin-top: 0.5rem; }
.mt-4 { margin-top: 1rem; }
.mt-6 { margin-top: 1.5rem; }

.form-group { margin-bottom: 1.25rem; display: flex; flex-direction: column; }
.form-group label { margin-bottom: 0.4rem; font-size: 0.85rem; font-weight: 500; color: var(--text-secondary); }
textarea.input-field { resize: none; min-height: 80px; }

.w-full { width: 100%; }
.mt-auto { margin-top: auto; }
.pt-4 { padding-top: 1rem; }
.mt-8 { margin-top: 3rem; }
.text-center { text-align: center; }

.gallery-container { width: 100%; border-top: 1px solid var(--border-color); padding-top: 2rem; }
.image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }
.image-wrap { position: relative; border-radius: var(--radius-sm); overflow: hidden; aspect-ratio: 4/5; background: var(--bg-color); border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); }
.image-wrap img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.image-wrap:hover img { transform: scale(1.03); }

.overlay {
  position: absolute; inset: 0; background: linear-gradient(to top, rgba(0,0,0,0.6) 0%, transparent 40%);
  display: flex; align-items: flex-end; justify-content: center; padding-bottom: 1.5rem;
  opacity: 0; transition: opacity 0.3s ease;
}
.image-wrap:hover .overlay { opacity: 1; }

.download-btn { background: var(--surface-color); color: var(--text-primary); border: none; padding: 0.6rem 1.25rem; border-radius: 20px; font-weight: 600; cursor: pointer; font-size: 0.85rem; box-shadow: var(--shadow-md); transition: transform 0.2s; }
.download-btn:hover { transform: scale(1.05); }

/* Left Panel Upload Field */
.upload-zone-border { border: 1.5px dashed var(--border-color); border-radius: var(--radius-sm); padding: 1.5rem; text-align: center; cursor: pointer; background: var(--bg-color); transition: all 0.2s ease; margin-top: 0; min-height: 150px;}
.upload-zone-border:hover { border-color: var(--accent-color); }
.upload-zone-border.has-file { border: 1.5px solid var(--success); background: rgba(16,185,129,0.05); }
.text-sm { font-size: 0.85rem; color: var(--text-muted); }
.success-text { color: var(--success); font-weight: 600; }
.empty-icon { width: 32px; height: 32px; color: var(--text-muted); margin: 0 auto; }
</style>
