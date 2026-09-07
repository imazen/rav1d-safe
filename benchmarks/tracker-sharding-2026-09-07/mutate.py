from pathlib import Path
import subprocess,hashlib,json,os
source=Path('crates/rav1d-disjoint-mut/src/tracker_shard.rs')
root=Path('/home/lilith/tmp/rav1d-perf-solution-2026-09-07')
saved=source.read_bytes()
old=b'const SHARD_MIN_LEN: usize = 1024;'; mutant=b'const SHARD_MIN_LEN: usize = 64 * 1024;'
assert saved.count(old)==1
command=['cargo','nextest','run','-p','rav1d-disjoint-mut','--lib','-E','test(medium_shared_storage_spreads_simultaneous_row_guards)']
try:
 source.write_bytes(saved.replace(old,mutant))
 print('Testing restoration of the former threshold',flush=True)
 with (root/'mutation.log').open('w') as f:failed=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 assert failed.returncode!=0
 assert 'medium_shared_storage_spreads_simultaneous_row_guards' in (root/'mutation.log').read_text()
finally:
 source.write_bytes(saved)
assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(saved).digest()
with (root/'mutation-restored.log').open('w') as f:restored=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
assert restored.returncode==0
(root/'mutation.json').write_text(json.dumps(dict(command=command,mutant_exit=failed.returncode,restored_exit=restored.returncode,restored_sha256=hashlib.sha256(saved).hexdigest()),indent=2)+'\n')
print('Old threshold rejected; restored source passed',flush=True)
