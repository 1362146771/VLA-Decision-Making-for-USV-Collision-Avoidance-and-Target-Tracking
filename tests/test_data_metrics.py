from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from PIL import Image
from core.data import prepare, prepare_clip, EpisodeDataset, safe_path
from core.actions import ActionExecutor, ACTION_NAMES, ACTION_SCHEMA
from core.metrics import wilson,binary_auc,two_proportion_z
from core.rollouts import summarize_rollouts

class Tokenizer:
    def __call__(self,text,**kw):
        n=kw['max_length']; x=torch.tensor([[1,3,2]+[0]*(n-3)])
        return {'input_ids':x,'attention_mask':(x!=0).long()}

def episodes(root,cfg):
    rng=np.random.default_rng(77); records=[]
    for e in range(3):
        frames=[]
        for i in range(32):
            name=f'ep{e}/{i}.png'; p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rng.integers(0,255,(32,32,3),dtype=np.uint8)).save(p)
            frames.append({'image':name,'timestamp_s':i*cfg.frame_period_s,'action_id':(i//cfg.keyframe_stride)%6,
                           'collision':False,'caption':'vessel ahead','caption_mirrored':'vessel ahead'})
        records.append({'episode_id':f'ep{e}','task':'avoidance','action_schema':ACTION_SCHEMA,'initial_action_id':1,
                        'instruction':'turn starboard','instruction_mirrored':'turn port','frames':frames})
    p=root/'episodes.jsonl'; p.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    return p

def test_prepare_split_temporal_alignment_and_integrity(tmp_path,cfg):
    raw=episodes(tmp_path,cfg); output=tmp_path/'splits'
    summary=prepare(tmp_path,raw,output,cfg)
    assert set(summary)=={'train','val','test'}
    docs=[json.loads((output/f'{s}.json').read_text()) for s in summary]
    assert len(set(x for d in docs for x in d['episodes']))==3
    ds=EpisodeDataset(tmp_path,output/'train.json',cfg,'world')
    r=ds.rows[0]; sample=ds[0]
    assert sample['images'].shape==(8,3,32,32)
    indices=[int(x.split('/')[-1].split('.')[0]) for x in r['world_images']]
    assert np.diff(indices).tolist()==[cfg.keyframe_stride]*7
    policy=EpisodeDataset(tmp_path,output/'test.json',cfg,'policy',Tokenizer())
    assert policy[0]['last_action']==1
    assert len(policy)==summary['test']['policy_samples']
    with pytest.raises(ValueError): EpisodeDataset(tmp_path,output/'test.json',cfg,'policy',Tokenizer(),augment=True)
    with pytest.raises(FileExistsError): prepare(tmp_path,raw,output,cfg)
    victim=tmp_path/policy.rows[0]['history'][-1]; victim.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): EpisodeDataset(tmp_path,output/'test.json',cfg,'policy',Tokenizer())

def test_no_escape(tmp_path):
    with pytest.raises(ValueError): safe_path(tmp_path,'../outside.png')

def test_caption_corpus_split_and_stage_guard(tmp_path,cfg):
    raw=episodes(tmp_path,cfg)
    eps=[json.loads(x) for x in raw.read_text().splitlines()]
    rows=[{'group_id':e['episode_id'],'image':e['frames'][0]['image'],
           'caption':'ship','caption_mirrored':'ship','source':'test synthetic','license':'test only',
           'source_kind':'synthetic','annotation_method':'programmatic'} for e in eps]
    pairs=tmp_path/'pairs.jsonl'; pairs.write_text('\n'.join(json.dumps(r) for r in rows))
    prepare_clip(tmp_path,pairs,tmp_path/'captions',cfg)
    ds=EpisodeDataset(tmp_path,tmp_path/'captions/train.json',cfg,'clip',Tokenizer(),augment=True)
    assert ds[0]['images'].shape==(3,32,32)
    with pytest.raises(ValueError,match='Caption-only'): EpisodeDataset(tmp_path,tmp_path/'captions/train.json',cfg,'world')

def test_action_decode_and_increment_rate():
    e=ActionExecutor(); assert ACTION_NAMES[1]=='STOP'
    e.accept(1); assert e.update()['throttle']==0
    e.accept(4)
    for _ in range(5): e.update()
    assert e.throttle==.1
    assert e.update(radar_enabled=True,radar_distance_m=25)['throttle']==0
    with pytest.raises(ValueError): e.accept(99)
    with pytest.raises(ValueError): e.update(radar_enabled=True)

def test_metrics():
    assert .69 < wilson(41,50)[0] < .70
    assert binary_auc([1,2,3,4],[0,0,1,1])==1
    assert binary_auc([1,1],[0,1])==.5
    assert binary_auc([1,2],[1,1]) is None
    assert two_proportion_z(41,50,41,50)['p_two_sided']==1

def test_rollout_success_is_stricter_than_collision_free(tmp_path):
    records=[]
    for i,distance in enumerate([19.,30.]):
        records.append({'trial_id':str(i),'checkpoint_sha256':'a'*64,'seed':i,'mode':'policy_only',
                        'task':'avoidance','initial_distance_m':100,'completed':True,
                        'trace':[{'timestamp_s':0.,'obstacle_distance_m':distance,'pursuit_progress_m':0.,
                                  'collision':False,'inside_navigable_area':True,'target_visible':True}]})
    p=tmp_path/'trials.jsonl'; p.write_text('\n'.join(json.dumps(r) for r in records))
    result=summarize_rollouts(p,tmp_path/'result.json',2)['groups'][0]
    assert result['sr']==.5 and result['collision_free']==1 and result['near_miss']==.5
    with pytest.raises(ValueError,match='expected 50'): summarize_rollouts(p,tmp_path/'invalid.json')
