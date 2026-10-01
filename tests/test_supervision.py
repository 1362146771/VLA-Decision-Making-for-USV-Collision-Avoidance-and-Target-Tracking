import json
from pathlib import Path
from dataclasses import asdict

import pytest
import torch
import yaml

from core.data import prepare, EpisodeDataset
from core.model import OceanVLA
from core.pipeline import preflight
from core.supervision import validate_caption
from test_data_metrics import episodes, Tokenizer


def test_collision_target_uses_recorded_future_events(tmp_path, cfg):
    raw = episodes(tmp_path, cfg)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    for episode in records:
        episode['frames'][26]['collision'] = True
    raw.write_text('\n'.join(json.dumps(row) for row in records))
    prepare(tmp_path, raw, tmp_path/'split', cfg)
    ds = EpisodeDataset(tmp_path, tmp_path/'split/train.json', cfg, 'world')
    first = ds.rows[0]
    assert first['risk'] == 1
    assert first['world_actions'] == [0, 1, 2, 3, 4, 5, 0, 1]
    assert first['last_action'] == 2 and first['target'] == 3
    for episode in records:
        episode['frames'][26]['collision'] = False
    raw.write_text('\n'.join(json.dumps(row) for row in records))
    prepare(tmp_path, raw, tmp_path/'safe_split', cfg)
    safe = EpisodeDataset(tmp_path, tmp_path/'safe_split/train.json', cfg, 'world')
    assert all(row['risk'] == 0 for row in safe.rows)


@pytest.mark.parametrize('field', ['pseudo_action_id', 'executed_proxy', 'risk_proxy'])
def test_proxy_records_rejected_before_preparation(tmp_path, cfg, field):
    raw = episodes(tmp_path, cfg)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    records[0]['frames'][0][field] = 0
    raw.write_text('\n'.join(json.dumps(row) for row in records))
    with pytest.raises(ValueError, match='Recorded expert'):
        prepare(tmp_path, raw, tmp_path/'split', cfg)


def test_caption_annotation_source_contract():
    row = {'caption':'ship', 'caption_mirrored':'ship', 'source':'SeaDronesSee',
           'license':'test', 'source_kind':'public', 'annotation_method':'human'}
    validate_caption(row)
    with pytest.raises(ValueError, match='human annotation'):
        validate_caption({**row, 'annotation_method':'programmatic'})
    validate_caption({**row, 'source_kind':'synthetic', 'annotation_method':'programmatic'})


def test_clip_cannot_fall_back_to_episode_captions(tmp_path, cfg):
    prepare(tmp_path, episodes(tmp_path, cfg), tmp_path/'split', cfg)
    with pytest.raises(ValueError, match='separate maritime caption corpus'):
        EpisodeDataset(tmp_path, tmp_path/'split/train.json', cfg, 'clip', Tokenizer())


@pytest.mark.parametrize('public,synthetic,ready', [(7000,8000,True), (0,15000,False), (7000,7999,False)])
def test_caption_composition_preflight(tmp_path, cfg, public, synthetic, ready):
    opts = {'epochs':1,'lr':.001,'global_batch':2,'micro_batch':2,
            'weight_decay':0.,'amp':False,'balanced':False}
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'model':asdict(cfg), 'training':{
        'seed':1,'world':opts,'clip':opts,'policy':opts}}))
    for kind in ('episodes', 'captions'):
        folder = tmp_path/kind
        folder.mkdir()
        for index, split in enumerate(('train', 'val', 'test')):
            if kind == 'episodes':
                rows = [{'target':c,'last_action':0,'history':['x']*4,
                         'world_images':['x']*8,'world_actions':[c]*8,'risk':c%2,
                         'instruction_mirrored':'keep clear'} for c in range(6)]
            else:
                rows = []
                for source, count in (('public',public), ('synthetic',synthetic)):
                    n = count//3 + (index < count%3)
                    rows.extend({'source_kind':source, 'annotation_method':'human' if source=='public' else 'programmatic',
                                 'caption':'ship', 'caption_mirrored':'ship', 'source':source, 'license':'test'} for _ in range(n))
            doc = {'schema':'oceanvla.dataset.v1','split':split,'dataset_hash':kind,
                   'model_config':asdict(cfg),'episodes':[f'{kind}_{split}'],
                   'assets':{},'samples':rows}
            if kind == 'captions': doc['corpus_kind'] = 'caption_pairs'
            (folder/f'{split}.json').write_text(json.dumps(doc))
    result = preflight(config, tmp_path/'episodes', tmp_path/'captions')
    assert result['ready'] == ready, result['errors']


def test_each_history_frame_and_future_latents_affect_policy(cfg, hf_config, batch, monkeypatch):
    torch.manual_seed(19)
    model = OceanVLA(cfg, hf_config=hf_config, pretrained=False).eval()
    inputs = {k:v for k,v in batch.items() if k != 'targets'}
    with torch.no_grad():
        expected = model(**inputs)
        for frame in range(cfg.memory_frames):
            images = inputs['images'].clone()
            images[:,frame] = torch.randn_like(images[:,frame]) * 3
            assert not torch.allclose(expected, model(**{**inputs, 'images':images}), atol=1e-7, rtol=1e-7)
        rollout = model.world.rollout
        monkeypatch.setattr(model.world, 'rollout', lambda latent, actions: rollout(latent, actions) + 3)
        assert not torch.allclose(expected, model(**inputs), atol=1e-7, rtol=1e-7)


def test_default_launcher_uses_expert_pipeline():
    root = Path(__file__).resolve().parents[1]
    text = (root/'start_policy.sh').read_text()
    assert '-m core.pipeline' in text
    assert 'bootstrap' not in text
    assert '--episodes' in text and '--captions' in text
    assert yaml.safe_load((root/'configs/train.yaml').read_text()) == yaml.safe_load((root/'configs/train1.yaml').read_text())
