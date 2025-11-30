@echo off
REM Bitcoin Trading AI - Evaluation Script
REM Evaluates predictions 12 hours after generation

REM Change to project directory
cd /d C:\Apps\Obsidian\Jarl\crypto-ai-trader

REM Activate conda environment
call C:\Users\JarlJ\anaconda3\Scripts\activate.bat crypto-ai-trader

REM Run the evaluation framework
python scripts\evaluation_framework.py

REM Log completion
echo [%date% %time%] Evaluation completed >> evaluation.log
