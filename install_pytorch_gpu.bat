@echo off
chcp 65001 >nul
echo ========================================
echo 安装GPU版本的PyTorch
echo ========================================
echo.
echo 当前安装的是CPU版本的PyTorch，需要重新安装GPU版本
echo.

REM 首先卸载CPU版本
echo 正在卸载CPU版本的PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo 正在安装GPU版本的PyTorch (CUDA 12.4)...
echo 这可能需要几分钟时间，请耐心等待...
echo.

REM 安装CUDA 12.4版本的PyTorch（最新稳定版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo ========================================
echo 验证安装...
echo ========================================
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

echo.
if %ERRORLEVEL% EQU 0 (
    echo 安装完成！
    echo 如果显示 "CUDA available: True"，则可以开始训练
    echo 运行 train_v3.bat 开始训练
) else (
    echo 安装可能出现问题，请检查错误信息
)
echo ========================================
pause
