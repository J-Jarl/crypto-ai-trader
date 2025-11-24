@echo off
REM Bitcoin Trading AI - Automated Run Script
REM Activates conda environment and executes trading analysis

REM Change to project directory
cd /d C:\Apps\Obsidian\Jarl\crypto-ai-trader

REM Activate conda environment
call C:\Users\JarlJ\anaconda3\Scripts\activate.bat crypto-ai-trader

REM Run the trading AI
python scripts\trading_ai.py

REM Log completion
echo [%date% %time%] Trading AI execution completed >> automation.log


