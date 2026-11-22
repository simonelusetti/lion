"""Wait for the active experiment, then update only its report."""
import os,select,subprocess,sys
from pathlib import Path
from omegaconf import OmegaConf


def main():
    for row in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]:
        pid,command=row.strip().split(maxsplit=1);args=command.split()
        if Path(args[0]).name not in ('python','python3','python3.10') or '-m' not in args:continue
        if args[args.index('-m')+1]!='lion.augmentation.run':continue
        try:fd=os.pidfd_open(int(pid))
        except ProcessLookupError:continue
        print(f'Waiting for augmentation process {pid}; training and generation continue unchanged.',flush=True)
        select.select([fd],[],[]);os.close(fd)
    cfg=OmegaConf.load(sys.argv[1]);cfg.stage.experiment='report'
    from .run import main as report_main
    report_main(cfg)


if __name__=='__main__':main()
