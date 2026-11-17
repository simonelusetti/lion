"""Wait for the already-running Stage 1, then replace this process with study 4."""
import json
import os
import select
import subprocess
import sys
from ..diagnostics.common import ROOT


def main():
    rows=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]
    for row in rows:
        pid,command=row.strip().split(maxsplit=1);args=command.split()
        if '-m' not in args:continue
        if args[args.index('-m')+1]!='lion.invariance.stage1':continue
        try:fd=os.pidfd_open(int(pid))
        except ProcessLookupError:continue
        print(f'Waiting for Stage 1 process {pid}; study 4 has not started.',flush=True)
        select.select([fd],[],[]);os.close(fd)
    timing=json.loads((ROOT/'results'/'invariance_stage1'/'timing.json').read_text())
    if 'ended_unix' not in timing:raise RuntimeError('Stage 1 exited before its final report; inspect it before starting study 4.')
    print(f"Stage 1 completed in {timing['wall_clock_seconds']/3600:.2f} hours. Starting study 4.",flush=True)
    os.execv(sys.executable,[sys.executable,'-B','-u','-m','lion.zfree.run',*sys.argv[1:]])


if __name__=='__main__':main()
