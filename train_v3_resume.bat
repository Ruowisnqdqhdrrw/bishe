@echo off
chcp 65001 >nul
REM 从检查点恢复训练
REM 
REM 使用场景：
REM 1. 训练被Ctrl+C中断后继续
REM 2. 训练因其他原因中断后继续
REM 3. 想要继续训练更多轮次

echo ========================================
echo 从检查点恢复训练
echo ========================================
echo.
echo 提示：按 Ctrl+C 可以安全中断训练并保存检查点
echo.

python -m train.train_e2e_v3 ^
    --data_roots "D:/dataset/crohme_2019_rendered_latex" "D:/dataset/HME100K_prepared" ^
    --train_splits train ^
    --valid_splits valid ^
    --test_splits test ^
    --epochs 200 ^
    --batch_size 16 ^
    --lr 1e-4 ^
    --min_lr 1e-6 ^
    --weight_decay 0.05 ^
    --device cuda ^
    --amp ^
    --num_workers 4 ^
    --img_height 64 ^
    --max_width 512 ^
    --encoder_dim 512 ^
    --decoder_dim 512 ^
    --embed_dim 256 ^
    --attention_dim 256 ^
    --num_heads 8 ^
    --decoder_layers 3 ^
    --dropout 0.4 ^
    --encoder_dropout 0.2 ^
    --drop_path 0.1 ^
    --coverage_weight 0.1 ^
    --use_se ^
    --teacher_forcing 1.0 ^
    --grad_clip 2.0 ^
    --label_smoothing 0.1 ^
    --grad_accum 2 ^
    --scheduler cosine ^
    --warmup_epochs 10 ^
    --patience 35 ^
    --augment_level strong ^
    --checkpoint_dir checkpoints_e2e_v3_mixed ^
    --resume

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo 训练完成！
) else (
    echo 训练已中断，检查点已保存
    echo 再次运行此脚本继续训练
)
echo ========================================
pause
