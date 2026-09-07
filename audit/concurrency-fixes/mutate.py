import pathlib, subprocess, json, os, time
root=pathlib.Path('/home/lilith/work/zen/rav1d-safe'); out=pathlib.Path('/home/lilith/tmp/rav1d-policy-mc-2026-09-07/validation')
picture=root/'include/dav1d/picture.rs'; mc=root/'src/safe_simd/mc_reference.rs'
originals={p:p.read_text() for p in [picture,mc]}
cmd=['cargo','nextest','run','--release','-p','rav1d-safe','--lib','-E','test(picture_policy) | test(reference_window)','--test-threads','1']
env=os.environ.copy();env['CARGO_TERM_COLOR']='never';env.pop('RUSTFLAGS',None)
records=[]
try:
 old='self.threading\n            .map_or_else(tile_threading_active, |p| p.parallel)'
 assert old in originals[picture]
 picture.write_text(originals[picture].replace(old,'tile_threading_active()'))
 old='slice_as::<_, BD::Pixel>(start..end)';assert old in originals[mc]
 mc.write_text(originals[mc].replace(old,'slice_as::<_, BD::Pixel>(start..end - 1)'))
 with (out/'mutation.log').open('w') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 log=(out/'mutation.log').read_text();assert p.returncode!=0
 for name in ['explicit_picture_policy_controls_gap_reservations_after_global_promotion','picture_policy_is_local_and_survives_decoder_lifetimes','mc_reference_windows_cover_all_filters_phases_widths_and_depths','reference_window_reserves_its_full_hull_including_gaps','warp_reference_windows_cover_both_strides_and_all_depths']:
  assert any('FAIL' in line and name in line for line in log.splitlines()),name
 records.append(dict(arm='mutant',exit=p.returncode,expected_failed_tests=5,mutations=['use process-global policy for every picture','omit last reserved source pixel'],command=cmd))
finally:
 for path,source in originals.items():path.write_text(source)
with (out/'mutation-restored.log').open('w') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
records.append(dict(arm='restored',exit=p.returncode,command=cmd));(out/'mutations.json').write_text(json.dumps(records,indent=2)+'\n')
assert p.returncode==0
print(json.dumps(records),flush=True)
