@echo off
echo ========================================
echo     Backstory Validation RAG Pipeline
echo ========================================
echo.

echo [Stage 1] Ingesting text from books...
python Pipeline\ingestion.py
if %errorlevel% neq 0 (
    echo ERROR: Stage 1 failed. Exiting.
    pause
    exit /b 1
)
echo [Stage 1] Done.
echo.

echo [Stage 2-4] Chunking...
python Pipeline\Chunking.py
if %errorlevel% neq 0 (
    echo ERROR: Stage 2-4 failed. Exiting.
    pause
    exit /b 1
)
echo [Stage 2-4] Done.
echo.

echo [Stage 5] Atomic chunking...
python Pipeline\atomicChunking.py
if %errorlevel% neq 0 (
    echo ERROR: Stage 5 failed. Exiting.
    pause
    exit /b 1
)
echo [Stage 5] Done.
echo.

echo [Stage 5b] Generating embeddings and building FAISS index...
python Pipeline\embeddingsGeneration.py
if %errorlevel% neq 0 (
    echo ERROR: Embeddings generation failed. Exiting.
    pause
    exit /b 1
)
echo [Stage 5b] Done.
echo.

echo [Stage 6-10] Running claim extraction, retrieval, verification, aggregation and explanation...
python Pipeline\explanationLayer.py
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed at explanation stage. Exiting.
    pause
    exit /b 1
)
echo [Stage 6-10] Done.
echo.

echo ========================================
echo     Pipeline completed successfully.
echo ========================================
pause
