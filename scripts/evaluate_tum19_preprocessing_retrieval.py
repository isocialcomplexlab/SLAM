#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from loop_closure.evaluation.tum19_retrieval import evaluate_mode,pair_mode_rows,summarize_mode_rows,summarize_paired_rows,validate_descriptor_arrays

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def require(c,m):
 if not c: raise RuntimeError(m)
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--frame-manifest',type=Path,required=True);p.add_argument('--geometry-targets',type=Path,required=True);p.add_argument('--groundtruth',type=Path,required=True);p.add_argument('--split-descriptors',type=Path,required=True);p.add_argument('--full-image-descriptors',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def load_manifest(path):
 frames=[];gallery=[];queries=[];splits={}
 with path.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   frame=int(r['frame_id']);frames.append(frame)
   if r['role']=='gallery': gallery.append(frame)
   elif r['role']=='query': queries.append(frame);splits[frame]=r['split']
   else: raise RuntimeError(f'bad role {r["role"]}')
 require(len(frames)==1247,'manifest must have 1247 frames');require(gallery==list(range(4,618)),'gallery mismatch');require(queries==list(range(7743,8376)),'query mismatch');return np.asarray(frames,dtype=np.int64),np.asarray(gallery,dtype=np.int64),splits
def load_gt(path):
 rows=[]
 with path.open(encoding='utf-8',errors='replace') as f:
  for line in f:
   s=line.strip()
   if s and not s.startswith('#'): rows.append([float(x) for x in s.split()])
 gt=np.asarray(rows,dtype=np.float64);require(gt.shape==(8380,8),f'gt shape {gt.shape}');return gt
def load_npz(path,dim,expected):
 with np.load(path,allow_pickle=False) as z: d=np.asarray(z['descriptors']);ids=np.asarray(z['frame_ids'])
 return validate_descriptor_arrays(d,ids,expected_dim=dim,expected_frame_ids=expected)
def load_nearest(path):
 out={}
 with path.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f): out[int(r['query_frame'])]=(int(r['nearest_reference_frame']),float(r['nearest_distance_m']))
 require(len(out)==633,'nearest rows mismatch');return out
def write_csv(path,rows,overwrite):
 if path.exists() and not overwrite: raise RuntimeError(f'output exists: {path}')
 require(rows,f'empty rows {path}')
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
def main():
 a=parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);expected,gallery,splits=load_manifest(a.frame_manifest.resolve());queries=np.asarray(sorted(splits),dtype=np.int64);gt=load_gt(a.groundtruth.resolve());frozen=load_nearest(a.geometry_targets.resolve());pos=gt[:,1:4];geometry=np.linalg.norm(pos[queries,None,:]-pos[gallery][None,:,:],axis=2);require(np.isfinite(geometry).all(),'nonfinite geometry');exact={}
 for i,q in enumerate(queries):
  order=np.lexsort((gallery,geometry[i]));nf=int(gallery[order[0]]);nd=float(geometry[i,order[0]]);ff,fd=frozen[int(q)];require(nf==ff,f'nearest frame mismatch {q}');require(abs(nd-fd)<=1e-9,f'nearest distance mismatch {q}');exact[int(q)]=nf
 sd,si=load_npz(a.split_descriptors.resolve(),4096,expected);fd,fi=load_npz(a.full_image_descriptors.resolve(),2048,expected)
 sr=evaluate_mode(mode='split_left_right',descriptors=sd,frame_ids=si,gallery_frame_ids=gallery,query_frame_ids=queries,query_splits=splits,geometry_distances_m=geometry,exact_nearest_reference_frames=exact,expected_dim=4096)
 fr=evaluate_mode(mode='full_image',descriptors=fd,frame_ids=fi,gallery_frame_ids=gallery,query_frame_ids=queries,query_splits=splits,geometry_distances_m=geometry,exact_nearest_reference_frames=exact,expected_dim=2048)
 paired=pair_mode_rows(sr,fr);pq=out/'tum19_preprocessing_retrieval_per_query.csv';pp=out/'tum19_preprocessing_retrieval_paired.csv';sj=out/'tum19_preprocessing_retrieval_summary.json'
 for p in (pq,pp,sj):
  if p.exists() and not a.overwrite: raise RuntimeError(f'output exists: {p}')
 write_csv(pq,sr+fr,a.overwrite);write_csv(pp,paired,a.overwrite);summary={'status':'passed','binary_gt_threshold_used':False,'gallery_frames':[4,617],'query_frames':[7743,8375],'query_count':633,'split_left_right':summarize_mode_rows(sr),'full_image':summarize_mode_rows(fr),'paired_full_minus_split':summarize_paired_rows(paired),'inputs':{'frame_manifest_sha256':sha256(a.frame_manifest.resolve()),'geometry_targets_sha256':sha256(a.geometry_targets.resolve()),'groundtruth_sha256':sha256(a.groundtruth.resolve()),'split_descriptors_sha256':sha256(a.split_descriptors.resolve()),'full_image_descriptors_sha256':sha256(a.full_image_descriptors.resolve())}};sj.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
