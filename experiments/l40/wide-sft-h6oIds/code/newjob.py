import os,sys,json,subprocess
from pathlib import Path
w=Path(os.environ['WB']); d=Path(os.environ['BASE']); env=dict(os.environ);env.update(json.loads((w/'runtime.json').read_text()));env['PYTHONPATH']=str(w)+':'+env['PYTHONPATH'];env['MUJOCO_GL']='egl';env['OMP_NUM_THREADS']='2'
with (d/(sys.argv[1]+'.log')).open('a') as log:
 p=subprocess.Popen([env['PY'],'-u','-m','examples.so101_insertion.'+sys.argv[2],*sys.argv[3:]],cwd=w,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
