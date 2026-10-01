from dataclasses import asdict
from types import SimpleNamespace
import yaml
import pytest
import torch
from core import engine
from core.data import prepare
from core.checkpoint import restore
from test_data_metrics import episodes

def test_epoch_boundary_resume_matches_uninterrupted(tmp_path,cfg,monkeypatch):
    option={'epochs':2,'lr':.001,'global_batch':2,'micro_batch':1,'weight_decay':0.,'amp':False,'balanced':False}
    conf={'model':asdict(cfg),'training':{'seed':31,'world':option,'clip':{**option,'micro_batch':2},'policy':option}}
    cp=tmp_path/'config.yaml'; cp.write_text(yaml.safe_dump(conf))
    prepare(tmp_path,episodes(tmp_path,cfg),tmp_path/'split',cfg)
    def args(name,resume=None):
        return SimpleNamespace(config=cp,stage='world',output=tmp_path/name,device='cpu',resume=resume,
                               data_root=tmp_path,train_manifest=tmp_path/'split/train.json',val_manifest=tmp_path/'split/val.json')
    engine.train(args('full'))
    save=engine.save_checkpoint
    def stop_after_save(*a,**kw):
        save(*a,**kw)
        raise RuntimeError('test interruption after completed epoch')
    with monkeypatch.context() as m:
        m.setattr(engine,'save_checkpoint',stop_after_save)
        with pytest.raises(RuntimeError,match='test interruption'): engine.train(args('interrupted'))
    engine.train(args('resumed',tmp_path/'interrupted/epoch_001.pt'))
    full, fc=restore(tmp_path/'full/epoch_002.pt','world')
    resumed, rc=restore(tmp_path/'resumed/epoch_002.pt','world')
    assert fc['training_state']['global_step']==rc['training_state']['global_step']
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items())
