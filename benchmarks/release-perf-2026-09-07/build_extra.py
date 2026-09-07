import json,os,subprocess,time,shutil,hashlib
from pathlib import Path
root=Path(__file__).parent
records=[]
env=os.environ.copy();env['RUSTFLAGS']='-C llvm-args=-align-all-functions=4';env['TMPDIR']='/home/lilith/tmp'
for name,feature in [('restored','asm'),('latest',''),('latest','unchecked'),('latest','asm')]:
 arm=name+'-'+(feature or 'checked'); driver=root/(name+'-driver')
 cmd=['cargo','+stable','build','--manifest-path',str(driver/'Cargo.toml'),'--release']
 if feature:cmd+=['--locked','--features',feature]
 start=time.monotonic()
 with (root/(arm+'-build.log')).open('w') as f:
  r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=360)
 record=dict(arm=arm,command=cmd,exit=r.returncode,seconds=time.monotonic()-start)
 if r.returncode==0:
  shutil.copyfile(driver/'target/release'/('rav1d-release-profile-'+name),root/'bin'/arm)
  (root/'bin'/arm).chmod(0o755)
  record['sha256']=hashlib.sha256((root/'bin'/arm).read_bytes()).hexdigest()
 else:record['errors']=[x for x in (root/(arm+'-build.log')).read_text().splitlines() if 'error' in x][:16]
 records.append(record);(root/'extra-builds.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps(record),flush=True)
