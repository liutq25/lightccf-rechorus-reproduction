"""Rebuild the follow-up from this machine's completed earlier experiment matrix."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    for name in ('lock_followup_inputs.py','run_followup_matrix.py','summarize_followup.py'):
        subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
