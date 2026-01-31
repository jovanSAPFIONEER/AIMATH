@echo off
REM AIMATH Command Line Tool for Windows
REM Usage: aimath.bat solve "x^2 - 4 = 0"

python -m aimath.cli %*
