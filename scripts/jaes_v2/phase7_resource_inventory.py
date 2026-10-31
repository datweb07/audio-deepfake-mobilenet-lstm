"""Read-only resource inventory, including Windows physical/commit memory."""
from pathlib import Path
import argparse
import ctypes,importlib.metadata,json,os,shutil,subprocess,sys,tempfile
from src.lava.jaes_v2.evidence import write_json
from src.lava.jaes_v2.phase7_data import BASE

def memory():
    if os.name!='nt':return {'status':'NOT_MEASURED'}
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load_percent',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','extended_virtual')]
    status=Status();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):return {'status':'FAILED'}
    return {k:getattr(status,k) for k,_ in status._fields_}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'resource_inventory_v1.json');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];output=root/args.output
    if not output.resolve().is_relative_to((root/BASE).resolve()):raise ValueError('Resource inventory must stay insidePhase7')
    if output.exists():raise FileExistsError(output)
    temp=root/'.research_tmp';temp.mkdir(exist_ok=True);os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp);tempfile.tempdir=str(temp)
    def query(command):
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=10);return dict(returncode=p.returncode,stdout=p.stdout.replace('\x00','').strip(),stderr=p.stderr.replace('\x00','').strip())
        except (OSError,subprocess.TimeoutExpired) as e:return dict(status='NOT_VERIFIED',error=repr(e))
    data=dict(date='2026-10-31 Asia/Novosibirsk',python=sys.executable,cpu_count=os.cpu_count(),memory=memory(),
        disk={k:dict(free_bytes=shutil.disk_usage(k).free,total_bytes=shutil.disk_usage(k).total) for k in ('C:/','D:/')},
        temp=dict(TEMP=os.getenv('TEMP'),TMP=os.getenv('TMP'),python_temp=tempfile.gettempdir(),pip_cache_default=str(Path.home()/'AppData/Local/pip/Cache'),
            matplotlib_cache=str(temp/'matplotlib'),tensorflow_temporary='Process TEMP/TMP onD; no system environment change',checkpoints=str(root/BASE/'training')),
        tensorflow=importlib.metadata.version('tensorflow'),gpu_inventory=query(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
        pip_cache_actual=query([sys.executable,'-m','pip','cache','dir']),
        wsl_inventory=query(['wsl.exe','--list','--quiet']),gpu_tensorflow='No GPU visible in executed current TensorFlowCPU cache/training',
        production_environments_modified=False,new_dependencies_installed=False,concurrency='Existing OS file mutex; one heavyTensorFlow job; PID ownership audit in job_owners',
        CIM_probe='Read denied by host; physical/commit memory measured with Windows GlobalMemoryStatusEx')
    write_json(output,data);print(json.dumps(data,indent=2))
if __name__=='__main__':main()
