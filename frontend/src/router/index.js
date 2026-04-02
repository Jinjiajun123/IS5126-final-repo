import { createRouter, createWebHistory } from 'vue-router'
import Diagnostic from '../views/Diagnostic.vue'
import Generator from '../views/Generator.vue'

const routes = [
  {
    path: '/',
    name: 'Diagnostic',
    component: Diagnostic
  },
  {
    path: '/generator',
    name: 'Generator',
    component: Generator
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
