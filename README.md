# NexusAnalytics - Merchant AI Suite

An AI-powered e-commerce analytics platform that combines machine learning prediction, LLM-based diagnostics, and AI image generation to help merchants optimize product listings and marketing strategies.

## Features

The platform consists of three core modules:

- **Analytics Hub** - Dual-engine product evaluation combining ML purchase prediction with LLM-powered diagnostic reports (strengths, weaknesses, persona analysis). Includes a live Taobao-style product preview.
- **Ad Studio** - AI-powered advertising image generator using Doubao Seedream model. Supports style presets (minimalist, lifestyle, luxury) and custom keywords.
- **User Insights** - Session-based user intent analysis with simulated user behavior. Provides purchase probability distribution, intent tier segmentation, and actionable recommendations (coupon targeting, traffic push, etc.).

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git LFS (for large model/data files)

## Setup

1. Clone the repository and pull LFS files:

```bash
git clone <repo-url>
cd "Final Project"
git lfs pull
```

2. Create and activate Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
```

3. Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

4. Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ARK_API_KEY` | Volcengine ARK API key for Doubao LLM and image generation | Optional (has default) |

## Quick Start

Run both backend and frontend with a single command:

```bash
chmod +x start.sh
./start.sh
```

Or start them separately:

```bash
# Backend (port 8000)
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend (port 5173)
cd frontend && npm run dev
```

Once running, open http://localhost:5173 in your browser.

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI backend (API endpoints, ML pipeline, LLM integration)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── views/           # Vue page components (Diagnostic, Generator, AudienceInsights)
│   │   ├── api.js           # API client
│   │   ├── router/          # Vue Router config
│   │   └── App.vue          # Root component
│   ├── public/              # Static assets
│   └── package.json
├── training_pipeline.py     # ML model training code
├── model_artifacts.pkl      # Trained model artifacts (Git LFS)
├── model_metrics.json       # Model evaluation metrics
├── processed_data.csv       # Processed dataset (Git LFS)
├── social_ecommerce_data.csv # Raw dataset (Git LFS)
├── start.sh                 # One-click startup script
└── docs/                    # Documentation
```

## Tech Stack

- **Frontend**: Vue 3, Vue Router, ECharts, Axios
- **Backend**: FastAPI, scikit-learn, XGBoost, Pandas
- **AI Models**: Doubao Seed (LLM analysis), Doubao Seedream (image generation)
