from __future__ import annotations
import inspect
import numpy as np
import pytest
import loop_closure.evaluation.tum19_retrieval as retrieval
from loop_closure.evaluation.tum19_retrieval import corrected_l2_distance_matrix,evaluate_mode,pair_mode_rows,rank_gallery,summarize_mode_rows,validate_descriptor_arrays

def test_validate_descriptor_arrays_requires_dimension_and_exact_order():
 d=np.zeros((3,4),dtype=np.float32);f=np.array([4,5,6],dtype=np.int64)
 o,of=validate_descriptor_arrays(d,f,expected_dim=4,expected_frame_ids=[4,5,6]);assert o.shape==(3,4);assert of.tolist()==[4,5,6]
 with pytest.raises(ValueError,match='dimension'): validate_descriptor_arrays(d,f,expected_dim=5)
 with pytest.raises(ValueError,match='ordem'): validate_descriptor_arrays(d,f,expected_dim=4,expected_frame_ids=[4,6,5])

def test_corrected_l2_distance_matrix():
 q=np.array([[0.,0.]],dtype=np.float32);g=np.array([[3.,4.],[0.,1.]],dtype=np.float32);x=corrected_l2_distance_matrix(q,g);assert x.shape==(1,2);assert x[0].tolist()==pytest.approx([5.,1.])

def test_rank_gallery_tie_breaks_by_lower_reference_frame_id():
 d=np.array([1.,1.,.5],dtype=np.float32);f=np.array([20,10,30],dtype=np.int64);order=rank_gallery(d,f);assert f[order].tolist()==[30,10,20]

def test_evaluate_mode_spatial_errors_and_exact_nearest_rank():
 frames=np.array([10,20,30,100],dtype=np.int64);d=np.array([[0.,0.],[10.,0.],[20.,0.],[18.,0.]],dtype=np.float32);geometry=np.array([[.30,.10,.20]],dtype=np.float64)
 row=evaluate_mode(mode='synthetic',descriptors=d,frame_ids=frames,gallery_frame_ids=[10,20,30],query_frame_ids=[100],query_splits={100:'test'},geometry_distances_m=geometry,exact_nearest_reference_frames={100:20},expected_dim=2)[0]
 assert row['descriptor_top1_reference_frame']==30;assert row['top1_spatial_error_m']==pytest.approx(.20);assert row['best_spatial_error_at_5_m']==pytest.approx(.10);assert row['best_spatial_error_at_10_m']==pytest.approx(.10);assert row['exact_nearest_rank']==2;assert row['reciprocal_rank']==pytest.approx(.5);assert row['recall_at_1']==0;assert row['recall_at_5']==1;assert row['recall_at_10']==1

def test_mode_summary_reports_primary_and_secondary_metrics():
 rows=[{'query_frame':1,'split':'test','mode':'m','descriptor_top1_reference_frame':10,'top1_spatial_error_m':.1,'best_spatial_error_at_5_m':.05,'best_spatial_error_at_10_m':.02,'exact_nearest_reference_frame':11,'exact_nearest_rank':2,'reciprocal_rank':.5,'recall_at_1':0,'recall_at_5':1,'recall_at_10':1},{'query_frame':2,'split':'test','mode':'m','descriptor_top1_reference_frame':11,'top1_spatial_error_m':.3,'best_spatial_error_at_5_m':.10,'best_spatial_error_at_10_m':.04,'exact_nearest_reference_frame':11,'exact_nearest_rank':1,'reciprocal_rank':1.,'recall_at_1':1,'recall_at_5':1,'recall_at_10':1}]
 s=summarize_mode_rows(rows);assert s['test']['top1_spatial_error_m']['mean']==pytest.approx(.2);assert s['test']['MRR_exact_nearest']==pytest.approx(.75);assert s['test']['Recall@1_exact_nearest']==pytest.approx(.5)

def test_pair_mode_rows_uses_full_minus_split_sign():
 base={'query_frame':100,'split':'test','exact_nearest_rank':2,'best_spatial_error_at_5_m':.10,'best_spatial_error_at_10_m':.08};split=[{**base,'top1_spatial_error_m':.20}];full=[{**base,'top1_spatial_error_m':.15,'best_spatial_error_at_5_m':.12,'best_spatial_error_at_10_m':.04,'exact_nearest_rank':1}];r=pair_mode_rows(split,full)[0];assert r['delta_top1_spatial_error_m_full_minus_split']==pytest.approx(-.05);assert r['delta_best_spatial_error_at_5_m_full_minus_split']==pytest.approx(.02);assert r['delta_best_spatial_error_at_10_m_full_minus_split']==pytest.approx(-.04);assert r['delta_exact_nearest_rank_full_minus_split']==-1

def test_public_retrieval_api_introduces_no_binary_threshold_parameter():
 funcs=[retrieval.validate_descriptor_arrays,retrieval.corrected_l2_distance_matrix,retrieval.rank_gallery,retrieval.evaluate_mode,retrieval.summarize_mode_rows,retrieval.pair_mode_rows,retrieval.summarize_paired_rows]
 for f in funcs: assert 'threshold' not in inspect.signature(f).parameters
