<template>
  <div class="app-layout">
    <!-- Sidebar Navigation -->
    <aside class="sidebar">
      <div class="brand">
        <h2>Nexus<span style="color: var(--accent-color)">Analytics</span></h2>
        <p>Merchant AI Suite</p>
      </div>
      <nav class="nav-links">
        <router-link to="/" class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
          Analysis
        </router-link>
        <router-link to="/generator" class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon">
            <path d="M12 3l1.912 5.813a2 2 0 0 0 1.275 1.275L21 12l-5.813 1.912a2 2 0 0 0-1.275 1.275L12 21l-1.912-5.813a2 2 0 0 0-1.275-1.275L3 12l5.813-1.912a2 2 0 0 0 1.275-1.275L12 3z"></path>
          </svg>
          Ad Studio
        </router-link>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<style scoped>
.app-layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 260px;
  background: var(--surface-color);
  border-right: 1px solid var(--border-color);
  padding: 2rem 1.5rem;
  display: flex;
  flex-direction: column;
  position: fixed;
  height: 100vh;
}

.brand {
  margin-bottom: 3rem;
}

.brand h2 {
  font-size: 1.5rem;
  letter-spacing: -0.03em;
}

.brand p {
  font-size: 0.85rem;
  margin-top: 0.2rem;
}

.nav-links {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.85rem 1rem;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  text-decoration: none;
  font-weight: 500;
  transition: all 0.2s ease;
}

.icon {
  width: 20px;
  height: 20px;
}

.nav-item:hover, .nav-item.router-link-active {
  background: var(--bg-color);
  color: var(--text-primary);
}

.nav-item.router-link-active {
  background: rgba(10, 132, 255, 0.08);
  color: var(--accent-color);
}

.main-content {
  flex: 1;
  margin-left: 260px;
  padding: 2rem 3rem;
}

/* Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
