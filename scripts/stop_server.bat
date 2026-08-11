@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_server.ps1" %*
