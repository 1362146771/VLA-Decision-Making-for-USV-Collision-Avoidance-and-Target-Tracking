import torch
import pytest
from core.model import OceanWorldModel, MaritimeCLIP, OceanVLA, focal_loss
from core.engine import stage_loss
from core.checkpoint import save_checkpoint, restore
from core.config import ModelConfig

def inputs(batch): return {k:v for k,v in batch.items() if k!='targets'}

def test_all_policy_inputs_and_projections_connected(cfg,hf_config,batch):
    torch.manual_seed(7)
    model=OceanVLA(cfg,hf_config=hf_config,pretrained=False).eval()
    batch['images'].requires_grad_()
    logits=model(**inputs(batch))
    assert logits.shape==(2,6)
    focal_loss(logits,batch['targets']).backward()
    assert model.visual_projection.weight.grad.abs().sum()>0
    assert model.text_projection.weight.grad.abs().sum()>0
    assert model.world_projection.weight.grad.abs().sum()>0
    assert model.action_token.grad.abs().sum()>0
    assert batch['images'].grad.abs().sum((0,2,3,4)).min()>0
    assert all(p.grad is None for p in model.world.parameters())
    original=logits.detach()
    with torch.no_grad():
        changed=dict(inputs(batch)); changed['images']=batch['images'].detach().clone(); changed['images'][:,0]+=1
        assert not torch.allclose(original,model(**changed))
        changed=dict(inputs(batch)); changed['input_ids']=batch['input_ids'].clone(); changed['input_ids'][:,1]=9
        assert not torch.allclose(original,model(**changed))
        changed=dict(inputs(batch)); changed['last_action']=torch.tensor([5,5])
        assert not torch.allclose(original,model(**changed),atol=1e-7,rtol=1e-7)

def test_world_gradients_and_future_supervision(cfg):
    model=OceanWorldModel(cfg)
    x=torch.randn(2,8,3,32,32); a=torch.randint(0,6,(2,8)); risk=torch.tensor([0.,1.])
    result=model(x,a,risk)
    result['loss'].backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
               for module in (model.cnn,model.gru,model.dynamics,model.risk,model.action_embedding))
    with torch.no_grad():
        changed=x.clone(); changed[:,-1]+=2
        assert not torch.allclose(result['mse_k'],model(changed,a,risk)['mse_k'])
    with pytest.raises(ValueError): model(x[:,:4],a,risk)

def test_real_clip_contrastive_backward(cfg,hf_config,batch):
    model=MaritimeCLIP(cfg,hf_config=hf_config,pretrained=False)
    result=model(batch['images'][:,-1],batch['input_ids'],batch['attention_mask'])
    result['loss'].backward()
    assert model.clip.logit_scale.grad is not None
    assert model.clip.text_model.encoder.layers[0].self_attn.q_proj.weight.grad.abs().sum()>0
    assert model.clip.vision_model.encoder.layers[0].self_attn.q_proj.weight.grad is None
    assert model.clip.vision_model.encoder.layers[-1].self_attn.q_proj.weight.grad.abs().sum()>0

def test_save_restore_equivalence_and_reject_corrupt(cfg,hf_config,batch,tmp_path):
    model=OceanVLA(cfg,hf_config=hf_config,pretrained=False).eval()
    expected=model(**inputs(batch)).detach()
    path=tmp_path/'policy.pt'
    save_checkpoint(path,model,'policy',{'test_only':True})
    restored,ck=restore(path,'policy'); restored.eval()
    assert torch.equal(expected,restored(**inputs(batch)))
    with pytest.raises(FileExistsError): save_checkpoint(path,model,'policy',{})
    with pytest.raises(ValueError): restore(path,'world')
    ck['model_state_dict'].pop('action_token'); torch.save(ck,tmp_path/'broken.pt')
    with pytest.raises(RuntimeError): restore(tmp_path/'broken.pt')

def test_focal_reference_value_and_gradient():
    x=torch.randn(4,6,requires_grad=True); y=torch.tensor([0,1,4,5])
    loss=focal_loss(x,y); grad=torch.autograd.grad(loss,x)[0]
    logp=x.log_softmax(-1).gather(1,y[:,None]).squeeze(1)
    ref=(-((1-logp.exp())**2)*logp).mean()
    assert torch.allclose(loss,ref)
    assert torch.allclose(grad,torch.autograd.grad(ref,x)[0])

def test_production_world_count():
    model=OceanWorldModel(ModelConfig())
    assert sum(p.numel() for p in model.parameters())==1769665
    with torch.no_grad():
        intermediate=model.cnn[:-1](torch.zeros(1,3,224,224))
    assert intermediate.shape==(1,256,14,14)

@pytest.mark.parametrize('stage',['world','clip','policy'])
def test_stage_optimizer_updates_expected_modules(stage,cfg,hf_config,batch):
    if stage=='world':
        model=OceanWorldModel(cfg)
        data={'images':torch.randn(2,8,3,32,32),'expert_actions':torch.randint(0,6,(2,8)),'risk_labels':torch.tensor([0.,1.])}
    elif stage=='clip':
        model=MaritimeCLIP(cfg,hf_config=hf_config,pretrained=False)
        data={**batch,'images':batch['images'][:,-1]}
    else:
        model=OceanVLA(cfg,hf_config=hf_config,pretrained=False); data=batch
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    stage_loss(model,stage,data)['loss'].backward(); opt.step()
    changed=[k for k,p in model.named_parameters() if not torch.equal(before[k],p)]
    assert changed
    if stage=='policy': assert not any(k.startswith('world.') for k in changed)
