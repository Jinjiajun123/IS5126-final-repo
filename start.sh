#!/bin/bash

cd "$(dirname "$0")"

echo "=========================================="
echo " NexusAnalytics - Merchant AI Suite       "
echo "=========================================="

echo "1. Installing backend dependencies..."
.venv/bin/pip install -q -r backend/requirements.txt

echo "2. Booting FastAPI Backend..."
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "3. Installing frontend dependencies..."
cd frontend
npm install --silent

echo "4. Booting Vue Frontend..."
npm run dev &
FRONTEND_PID=$!
cd ..

echo "=========================================="
echo " Everything is running!"
echo " Frontend: http://localhost:5173"
echo " Backend : http://localhost:8000"
echo "=========================================="
echo "Press [CTRL+C] to stop both servers."

trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" INT TERM
wait
