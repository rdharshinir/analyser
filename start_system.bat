@echo off
echo ========================================
echo DRUG DISCOVERY ANALYZER - STARTUP SCRIPT
echo ========================================

echo Starting backend server...
cd backend
start "Backend Server" cmd /k "python app.py"
timeout /t 3 /nobreak >nul

echo Starting frontend development server...
cd ..
cd frontend
start "Frontend Server" cmd /k "npm run dev"
timeout /t 5 /nobreak >nul

echo Starting DeepChem prediction service...
cd ..
cd model
start "Prediction Service" cmd /k "python minimal_example.py"

echo.
echo ========================================
echo SYSTEM STARTUP COMPLETE
echo ========================================
echo Backend API: http://localhost:5000
echo Frontend UI: http://localhost:5173  
echo Prediction Service: Ready
echo ========================================
echo Press any key to close this window...
pause >nul