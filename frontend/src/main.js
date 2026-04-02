import { createApp } from 'vue'
import './assets/style.css'
import App from './App.vue'
import router from './router'

import ECharts from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, GaugeChart, RadarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, TitleComponent, LegendComponent, PolarComponent } from 'echarts/components'

use([
  CanvasRenderer,
  BarChart, GaugeChart, RadarChart,
  GridComponent, TooltipComponent, TitleComponent, LegendComponent, PolarComponent
])

const app = createApp(App)
app.component('v-chart', ECharts)
app.use(router)
app.mount('#app')
