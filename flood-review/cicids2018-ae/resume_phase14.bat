@echo off
rem Resume the DAE-LR experiment (Phase 14) after an interruption, for seeds 42, 1 and 2, three windows in parallel.
rem Finished models are kept and skipped; only the missing ones are trained. Each window closes when its seed is done.
rem If a window shows "Application Control policy has blocked", Windows is blocking PyTorch: use Colab instead (see TEAMMATE_GUIDE.md).
cd /d "%~dp0"
if not exist results_daelr mkdir results_daelr
if not exist models_daelr mkdir models_daelr
for %%S in (42 1 2) do (
  start "phase14 seed %%S" cmd /c "python src\phase14_daelr.py --seed %%S --threads 4 >> results_daelr\log_seed%%S.txt 2>&1"
)
echo Started three windows. Progress: type results_daelr\log_seed42.txt (and seed1, seed2). When all finish, tell Claude.
