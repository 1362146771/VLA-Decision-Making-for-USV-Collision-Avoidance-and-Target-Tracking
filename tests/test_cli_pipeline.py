
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPTokenizerFast
from transformers.models.clip.tokenization_clip import bytes_to_unicode
from core.checkpoint import restore
from test_data_metrics import episodes

def run(*args):
    result=subprocess.run([sys.executable,'-m','core.cli',*map(str,args)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    return result.stdout

def test_complete_cli_training_evaluation_and_resume(tmp_path,cfg,hf_config):
    tiny=tmp_path/'tiny-clip'; tiny.mkdir()
    tokens=list(bytes_to_unicode().values())
    vocab={s:i for i,s in enumerate(tokens+[s+'</w>' for s in tokens]+['<|startoftext|>','<|endoftext|>'])}
    (tiny/'vocab.json').write_text(json.dumps(vocab),encoding='utf-8')
    (tiny/'merges.txt').write_text('#version: 0.2\n',encoding='utf-8')
    tok=CLIPTokenizerFast(vocab_file=str(tiny/'vocab.json'),merges_file=str(tiny/'merges.txt'))
    tok.save_pretrained(tiny)
    hf_config['text_config'].update(vocab_size=len(vocab),bos_token_id=vocab['<|startoftext|>'],
                                  eos_token_id=vocab['<|endoftext|>'],pad_token_id=vocab['<|endoftext|>'])
    CLIPModel(CLIPConfig.from_dict(hf_config)).save_pretrained(tiny)
    cfg.clip_id=str(tiny)
    opts=lambda n:{'epochs':n,'lr':.001,'global_batch':2,'micro_batch':2,'weight_decay':.01,'amp':False,'balanced':False}
    conf={'model':asdict(cfg),'training':{'seed':17,'world':opts(2),'clip':opts(1),'policy':opts(1)}}
    config=tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(conf),encoding='utf-8')
    raw=episodes(tmp_path,cfg); split=tmp_path/'split'
    run('prepare','--config',config,'--data-root',tmp_path,'--episodes',raw,'--output',split)
    common=['--config',config,'--data-root',tmp_path,'--train-manifest',split/'train.json',
            '--val-manifest',split/'val.json','--device','cpu']
    run('train','--stage','world',*common,'--output',tmp_path/'world')
    # Epoch 1 remains if best; use a controlled one-epoch launch with two-epoch
    # horizon via a test interruption in a separate unit test when needed.
    world=tmp_path/'world'/'epoch_002.pt'
    model, ck=restore(world,'world'); assert ck['training_state']['epoch_completed']==2
    pairs=[]
    rng=np.random.default_rng(88)
    for group in range(3):
        for frame in range(4):
            name=f'caption_{group}_{frame}.png'
            Image.fromarray(rng.integers(0,255,(32,32,3),dtype=np.uint8)).save(tmp_path/name)
            pairs.append({'group_id':f'caption_{group}','image':name,'caption':'vessel ahead',
                          'caption_mirrored':'vessel ahead','source':'unity','license':'test',
                          'source_kind':'synthetic','annotation_method':'programmatic'})
    pairs_file=tmp_path/'pairs.jsonl'
    pairs_file.write_text('\n'.join(json.dumps(row) for row in pairs),encoding='utf-8')
    captions=tmp_path/'captions'
    run('prepare-clip','--config',config,'--data-root',tmp_path,'--pairs',pairs_file,'--output',captions)
    run('train','--stage','clip','--config',config,'--data-root',tmp_path,
        '--train-manifest',captions/'train.json','--val-manifest',captions/'val.json',
        '--device','cpu','--output',tmp_path/'clip')
    clip=tmp_path/'clip'/'epoch_001.pt'
    run('train','--stage','policy',*common,'--world-checkpoint',tmp_path/'world/best.pt',
        '--clip-checkpoint',tmp_path/'clip/best.pt','--output',tmp_path/'policy')
    policy=tmp_path/'policy'/'epoch_001.pt'
    # Same official CLI accepts the tensor-only production export and sidecar.
    policy=tmp_path/'policy'/'best.pt'
    assert all(isinstance(v,torch.Tensor) for v in torch.load(policy,weights_only=True).values())
    run('inspect','--checkpoint',policy)
    evaluation=tmp_path/'evaluation.json'
    run('evaluate','--checkpoint',policy,'--data-root',tmp_path,'--manifest',split/'test.json',
        '--output',evaluation,'--device','cpu','--batch-size','3')
    ev=json.loads(evaluation.read_text()); prepared=json.loads((split/'test.json').read_text())
    assert ev['samples']==len(prepared['samples'])==len(ev['predictions'])
    row=prepared['samples'][0]
    req=tmp_path/'request.json'; req.write_text(json.dumps({'instruction':row['instruction'],
        'last_executed_action':row['last_action'],'frames':[{'image':row['history'][-1],'timestamp_s':0.}]}))
    prediction=json.loads(run('predict','--checkpoint',policy,'--request',req))
    assert prediction['action_id'] in range(6)
    from core.config import sha256_file, canonical_hash
    _, policy_record = restore(policy, "policy")
    architecture = tmp_path/"architecture.json"
    architecture.write_text(json.dumps(policy_record["clip_config"]))
    provenance = tmp_path/"provenance.json"
    provenance.write_text(json.dumps({"weights_sha256": sha256_file(policy), "metadata": policy_record["metadata"]}))
    partition = json.loads((split/"partition.json").read_text())
    documents = {name: json.loads((split/(name+".json")).read_text()) for name in ("train", "val", "test")}
    index = {"schema": "oceanvla.dataset_index.v1", "partition": "partition.json",
             "partition_sha256": canonical_hash(partition), "partition_file_sha256": sha256_file(split/"partition.json"),
             "episode_count": sum(len(d["episodes"]) for d in documents.values()),
             "frame_count": sum(sum(e["raw_frames"] for e in d["decision_support"].values()) for d in documents.values()),
             "usable_windows": sum(len(d["samples"]) for d in documents.values()), "splits": {}}
    for name, doc in documents.items():
        index["splits"][name] = {"manifest": name+".json", "manifest_sha256": sha256_file(split/(name+".json")),
                                 "decision_support_sha256": doc["decision_support_sha256"], "episodes": len(doc["episodes"]),
                                 "raw_frames": sum(e["raw_frames"] for e in doc["decision_support"].values()),
                                 "usable_windows": len(doc["samples"])}
    dataset = split/"dataset.json"
    dataset.write_text(json.dumps(index))
    pure = tmp_path/"weights.pt"
    torch.save(torch.load(policy, weights_only=True), pure)
    provenance.write_text(json.dumps({"weights_sha256": sha256_file(pure), "metadata": policy_record["metadata"]}))
    options = ["--checkpoint", pure, "--config", config, "--clip-config", architecture,
               "--tokenizer", tmp_path/"policy/tokenizer", "--device", "cpu"]
    new_prediction = json.loads(run("predict", *options, "--request", req))
    assert prediction == new_prediction
    run("predict", *options, "--dataset", dataset, "--data-root", tmp_path, "--output", tmp_path/"predictions.json")
    predicted = json.loads((tmp_path/"predictions.json").read_text())
    assert [r["top3"] for r in predicted["predictions"]] == [r["top3"] for r in ev["predictions"]]
    run("evaluate", *options, "--dataset", dataset, "--data-root", tmp_path,
        "--provenance", provenance, "--output", tmp_path/"indexed_metrics.json", "--batch-size", "3")
    indexed = json.loads((tmp_path/"indexed_metrics.json").read_text())
    assert indexed["top1"] == ev["top1"]
    assert indexed["predictions"] == ev["predictions"]
    failed = subprocess.run([sys.executable, "-m", "core.cli", "evaluate", *map(str, options),
                             "--dataset", str(dataset), "--data-root", str(tmp_path),
                             "--output", str(tmp_path/"missing.json")], capture_output=True)
    assert failed.returncode != 0 and b"evaluation_provenance" in failed.stderr
    assert not (tmp_path/"missing.json").exists()

    # Completed checkpoints cannot accidentally restart or overwrite a run.
    result=subprocess.run([sys.executable,'-m','core.cli','train','--stage','world',*map(str,common),
                           '--resume',str(world),'--output',str(tmp_path/'resume')],capture_output=True)
    assert result.returncode!=0 and b'already completed' in result.stderr
