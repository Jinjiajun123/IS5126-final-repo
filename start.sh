#!/bin/bash

echo "=========================================="
echo " Starting Vue Frontend and FastAPI Backend "
echo "=========================================="

echo "1. Installing backend dependencies..."
# Use the project's virtual environment
.venv/bin/pip install -r backend/requirements.txt

echo "2. Booting FastAPI Backend..."
# Run FastAPI on background port 8000
.venv/bin/python -m uvicorn backend.main:app --port 8000 &
BACKEND_PID=$!

echo "3. Booting Vue Frontend..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!

echo "=========================================="
echo " Everything is running!"
echo " Frontend: http://localhost:5173"
echo " Backend : http://localhost:8000"
echo "=========================================="
echo "Press [CTRL+C] to stop both servers."

# Wait for process exit
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID" INT
wait
