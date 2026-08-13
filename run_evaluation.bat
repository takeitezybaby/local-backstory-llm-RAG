@echo off
echo ==================================================
echo   BACKSTORY RAG - RAGAS EVALUATION PIPELINE
echo ==================================================
echo.

echo Step 1: Collecting pipeline trace data for evaluation claims...
python Pipeline\eval_runner.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during trace collection!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Step 2: Running RAGAS and classification accuracy metrics...
python Pipeline\ragas_evaluator.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during evaluation calculation!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Evaluation complete! Report saved to Data\eval_results.json
pause
