@echo off
echo Starting all services...

REM Start the model server
echo Starting Model Server...
start "Model Server" cmd /k "cd model && python run_model.py"

REM Start the backend server
echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python app.py"

REM Start the frontend server
echo Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo All services are starting in separate windows.
