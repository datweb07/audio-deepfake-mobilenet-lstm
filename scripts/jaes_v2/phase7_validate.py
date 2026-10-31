"""Run the actual expanded and selected legacy suites under the research mutex and Dtemp."""
import argparse,os,subprocess,sys,tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=Path(__file__).resolve().parents[2]
    from src.lava.jaes_v2.phase7_data import safe_path
    safe_path(root,str(args.output));temporary=root/'.research_tmp';temporary.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temporary);os.environ['TMP']=str(temporary);tempfile.tempdir=str(temporary)
    os.environ['MPLCONFIGDIR']=str(temporary/'matplotlib')
    from scripts.jaes_v2.phase6_experiment import resource_lock
    with resource_lock(root):
        subprocess.run([sys.executable,'-X','utf8','-m','scripts.jaes_v2.validate','--output',str(args.output)],cwd=root,check=True)
if __name__=='__main__':main()
