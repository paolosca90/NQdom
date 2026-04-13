@echo off
cd /d "C:\Users\Paolo\Desktop\NQ\NQdom"
start "" /B python -u run_p1_to_p7_multiday.py --resume >> pipeline_run.log 2>&1
