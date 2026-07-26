#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from loop_closure.descriptors.resnet101 import DEFAULT_RESNET101_WEIGHTS,create_resnet101_model,extract_resnet101_descriptor
from loop_closure.descriptors.resnet101_full_image import extract_resnet101_full_image_descriptor
EXPECTED_DIMS={"split_left_right":4096,"full_image":2048}
def sha256(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def require(c,m):
 if not c: raise RuntimeError(m)
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--frame-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--mode',choices=['split_left_right','full_image','both'],default='both');p.add_argument('--device',default='cpu');p.add_argument('--max-frames',type=int);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def load_manifest(path,max_frames):
 rows=[]
 with path.open(newline='',encoding='utf-8') as f:
  r=csv.DictReader(f); require(set(r.fieldnames or [])=={'frame_id','role','split','image_path'},f'bad manifest schema {r.fieldnames}')
  for x in r: rows.append({'frame_id':int(x['frame_id']),'role':x['role'],'split':x['split'],'image_path':Path(x['image_path'])})
 require(len(rows)==1247,f'manifest rows={len(rows)}'); require(len({x['frame_id'] for x in rows})==1247,'duplicate frames'); require(not [x for x in rows if not x['image_path'].is_file()],'missing images')
 if max_frames is not None: require(max_frames>0,'max-frames must be >0'); rows=rows[:max_frames]
 return rows
def extract_mode(mode,model,rows,device,out,overwrite,manifest_sha):
 dim=EXPECTED_DIMS[mode]; extractor=extract_resnet101_descriptor if mode=='split_left_right' else extract_resnet101_full_image_descriptor; desc=[]; t=time.perf_counter()
 for i,row in enumerate(rows,1):
  with Image.open(row['image_path']) as im: d=extractor(model,im.convert('RGB'),device=device)
  a=d.detach().cpu().numpy().astype(np.float32,copy=False).reshape(-1); require(a.shape==(dim,),f'{mode} shape={a.shape}'); require(np.isfinite(a).all(),f'{mode} nonfinite'); desc.append(a)
  if i==1 or i==len(rows) or i%100==0: print(f'[{mode}] {i}/{len(rows)} frame={row["frame_id"]}',flush=True)
 elapsed=time.perf_counter()-t; matrix=np.ascontiguousarray(np.stack(desc),dtype=np.float32); ids=np.asarray([r['frame_id'] for r in rows],dtype=np.int64); roles=np.asarray([r['role'] for r in rows],dtype='U16'); splits=np.asarray([r['split'] for r in rows],dtype='U16')
 npz=out/f'tum19_resnet101_{mode}_descriptors.npz'; report=out/f'tum19_resnet101_{mode}_descriptors_report.json'
 for p in (npz,report):
  if p.exists() and not overwrite: raise RuntimeError(f'output exists: {p}')
 np.savez_compressed(npz,descriptors=matrix,frame_ids=ids,roles=roles,splits=splits)
 payload={'status':'passed','preprocessing_mode':mode,'descriptor_dimension':dim,'descriptor_shape':list(matrix.shape),'descriptor_dtype':str(matrix.dtype),'frame_count':len(rows),'first_frame':int(ids[0]),'last_frame':int(ids[-1]),'device':device,'weights':str(DEFAULT_RESNET101_WEIGHTS),'frame_manifest_sha256':manifest_sha,'seconds':elapsed,'seconds_per_frame':elapsed/len(rows),'frames_per_second':len(rows)/elapsed if elapsed>0 else None,'npz':str(npz),'npz_sha256':sha256(npz)}
 report.write_text(json.dumps(payload,indent=2)+"\n"); return payload
def main():
 a=parse_args(); manifest=a.frame_manifest.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); rows=load_manifest(manifest,a.max_frames); modes=['split_left_right','full_image'] if a.mode=='both' else [a.mode]; torch.manual_seed(0);np.random.seed(0);model=create_resnet101_model(device=a.device,weights=DEFAULT_RESNET101_WEIGHTS);model.eval(); reports=[extract_mode(m,model,rows,a.device,out,a.overwrite,sha256(manifest)) for m in modes]; summary={'status':'passed','frame_manifest':str(manifest),'frame_manifest_sha256':sha256(manifest),'modes':reports}; p=out/'tum19_preprocessing_ablation_descriptor_summary.json';
 if p.exists() and not a.overwrite: raise RuntimeError(f'output exists: {p}')
 p.write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
