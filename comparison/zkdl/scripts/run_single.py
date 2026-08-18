#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIGS={'cora':'cora_full.cfg','citeseer':'citeseer_full.cfg','pubmed':'pubmed_full.cfg','ppi':'ppi_full.cfg','ogbn-arxiv':'ogbn_arxiv_full.cfg'}
def call(cmd):subprocess.run([str(x) for x in cmd],check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=CONFIGS,required=True);p.add_argument('--device',choices=['cpu','gpu'],required=True);p.add_argument('--project-root',type=Path,default=ROOT.parents[1]);p.add_argument('--output-root',type=Path,default=ROOT/'runs');p.add_argument('--pp',type=Path,default=ROOT/'artifacts'/'pp.bin');p.add_argument('--relation-input',type=Path);p.add_argument('--certified-bounds-id');a=p.parse_args()
 out=a.output_root.resolve();base=ROOT.resolve()
 if out!=base and base not in out.parents:raise SystemExit('output-root must stay inside comparison/zkdl')
 run=out/a.dataset/a.device/'warm';art=run/'formal_artifacts';art.mkdir(parents=True,exist_ok=True);preflight=art/'preflight.json'
 try:call(['python3',ROOT/'scripts'/'preflight.py','--config',ROOT/'configs'/CONFIGS[a.dataset],'--project-root',a.project_root,'--output',preflight])
 except subprocess.CalledProcessError as e:
  info=json.loads(preflight.read_text());(run/'result.json').write_text(json.dumps({'dataset':a.dataset,'device':a.device,'status':info['status'],'blocker':info['ledger']['failures'],'proof_generated':False,'verified':False},indent=2)+'\n');raise SystemExit(e.returncode)
 if a.relation_input is None:raise SystemExit('--relation-input is required after accepted preflight; it must be the canonical adapter output for this dataset')
 cpu_binary=ROOT/'bin'/'zkdl-formal-cpu';prover_binary=ROOT/'bin'/f'zkdl-formal-{a.device}';statement=art/'statement.bin';witness=art/'witness.private.bin';model=ROOT/'artifacts'/a.dataset/'model_registry.bin';secret=ROOT/'artifacts'/a.dataset/'model_opening.private.bin';proof=art/'proof.bin';pm=run/'proof_manifest.json';vm=run/'verifier_manifest.json'
 for binary in (cpu_binary,prover_binary):
  if not binary.is_file():raise SystemExit(f'missing formal binary: {binary}; run scripts/build.sh')
 call([cpu_binary,'compile','--input',a.relation_input,'--statement',statement,'--witness',witness])
 if not model.exists() or not secret.exists():
  if not a.certified_bounds_id:raise SystemExit('--certified-bounds-id is required for first model registration')
  call([cpu_binary,'register-model','--pp',a.pp,'--statement',statement,'--witness',witness,'--model-registry',model,'--model-secret',secret,'--certified-bounds-id',a.certified_bounds_id])
 call([prover_binary,'prove','--pp',a.pp,'--statement',statement,'--model-registry',model,'--model-secret',secret,'--witness',witness,'--proof',proof,'--manifest',pm,'--device',a.device])
 call([cpu_binary,'verify','--pp',a.pp,'--statement',statement,'--model-registry',model,'--proof',proof,'--manifest',vm,'--device',a.device])
 pr=json.loads(pm.read_text());vr=json.loads(vm.read_text());(run/'result.json').write_text(json.dumps({'dataset':a.dataset,'device':a.device,'status':'ok','proof_generated':pr['proof_generated'],'verified':vr['verified'],'proof_size_bytes':pr['proof_size_bytes'],'prove_time_ms':pr['prove_time_ms'],'verify_time_ms':vr['verify_time_ms']},indent=2)+'\n')
if __name__=='__main__':main()
