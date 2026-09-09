@echo off
set JAVA_HOME=C:\slm\tools\jdk-17.0.19+10
cd /d C:\slm\android_app\AgentRuntime\android
call gradlew.bat --no-daemon assembleRelease > C:\slm\android_app\build.log 2>&1
echo EXITCODE %ERRORLEVEL% >> C:\slm\android_app\build.log
