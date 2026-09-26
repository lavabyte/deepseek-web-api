@echo off
rem Out-of-the-box launcher for Windows.
rem
rem There is nothing to install: the server has zero npm dependencies and Node
rem strips the TypeScript types natively. This script only checks that a new
rem enough Node.js is present and then starts the server.
rem
rem Usage:
rem   start.cmd            (double-click, or run from a terminal)
setlocal EnableExtensions
cd /d "%~dp0"

set MIN_MAJOR=22
set MIN_MINOR=6

where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js is not installed.
  echo This server needs Node.js %MIN_MAJOR%.%MIN_MINOR% or newer ^(no npm packages required^).
  echo Download: https://nodejs.org/
  pause
  exit /b 1
)

for /f "delims=" %%v in ('node -p "process.versions.node"') do set NODE_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("%NODE_VERSION%") do (
  set MAJOR=%%a
  set MINOR=%%b
)

if %MAJOR% LSS %MIN_MAJOR% (
  echo ERROR: Node.js %NODE_VERSION% is too old.
  echo This server needs Node.js %MIN_MAJOR%.%MIN_MINOR% or newer.
  echo Download: https://nodejs.org/
  pause
  exit /b 1
)
if %MAJOR% EQU %MIN_MAJOR% if %MINOR% LSS %MIN_MINOR% (
  echo ERROR: Node.js %NODE_VERSION% is too old.
  echo This server needs Node.js %MIN_MAJOR%.%MIN_MINOR% or newer.
  echo Download: https://nodejs.org/
  pause
  exit /b 1
)

if not exist .env (
  echo Note: no .env found. Using built-in defaults ^(port 8787, host 127.0.0.1^).
  echo       Copy .env.example to .env to change settings.
)

echo Starting deepseek-web-api on Node.js %NODE_VERSION%...
echo OpenAI-compatible endpoint: http://127.0.0.1:8787/v1
echo.
node src\server.mjs

pause
