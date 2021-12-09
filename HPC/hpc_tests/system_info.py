import multiprocessing as mp
import platform, socket, re, uuid, json, psutil, logging
import os

def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('../..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info["cores"]=mp.cpu_count()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)

parsed = json.loads(getSystemInfo())
print(json.dumps(parsed, indent=4, sort_keys=True))

print(os.environ['SLURM_NTASKS_PER_NODE'])