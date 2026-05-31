# core/inference/plan.py
"""
Minimal single-step inference for OceanVLA.

Runs one forward pass through the policy and returns the greedy action
prediction and world model token predictions. For closed-loop rollouts,
call this function at 2 Hz and pass the output action to ActionExecutor.

Action space (|A| = 6):
  0: FORWARD  1: STOP  2: TURNLEFT  3: TURNRIGHT  4: ACCELERATE  5: DECELERATE
"""

from typing import Any, Dict, Optional

import torch


@torch.no_grad()
def plan_and_predict(
    model,
    tokenizer,
    text: str,
    image: torch.Tensor,            # [3, H, W] — preprocessed image
    actions: Optional[torch.Tensor] = None,  # [K, A_dim] — action context (zeros if None)
    h_action: int = 6,
    h_world: int = 32,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-step greedy inference.

    Performs one forward pass through the full OceanVLA pipeline and returns
    the most likely action token plus world model predictions.

    Args:
        model:     OceanVLA model instance (in eval mode).
        tokenizer: Tokenizer matching the backbone.
        text:      Natural-language task instruction (e.g., "Avoid the cargo vessel").
        image:     Preprocessed RGB image tensor [3, H, W].
        actions:   Optional action context tensor [K, A_dim]. Uses zeros if None.
        h_action:  Action sequence length K.
        h_world:   World token sequence length N.
        device:    'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        dict with:
          'action_tokens'  [K]  — predicted discrete action indices
          'action_name'    str  — name of the primary action (first token)
          'world_tokens'   [N]  — predicted VQ world model tokens
          'raw'            dict — full model output dict
    """
    from core.execution.action_executor import ACTION_NAMES

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    # Tokenize instruction
    enc = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids          = enc["input_ids"].to(device)
    text_attention_mask = enc["attention_mask"].to(device)

    # Image: [3, H, W] → [1, 3, H, W]
    image = image.unsqueeze(0).to(device)

    # Action context: [K, A_dim] → [1, K, A_dim]
    a_dim = getattr(model, "action_dim", 6)
    if actions is None:
        actions = torch.zeros((h_action, a_dim), dtype=torch.float32)
    actions = actions.unsqueeze(0).to(device)

    # Placeholder targets (ignored during inference)
    action_targets = torch.full((1, h_action), -100, dtype=torch.long, device=device)
    wm_targets     = torch.full((1, h_world),  -100, dtype=torch.long, device=device)

    # Extend attention mask for non-text tokens
    B, L_txt = input_ids.shape
    extra = torch.ones(B, 1 + h_action + h_world, dtype=torch.long, device=device)
    text_attention_mask = torch.cat([text_attention_mask, extra], dim=1)

    out = model(
        input_ids=input_ids,
        text_attention_mask=text_attention_mask,
        images=image,
        actions=actions,
        action_targets=action_targets,
        wm_targets=wm_targets,
        lambda_action=1.0,
        lambda_world=0.0,  # Skip world model loss at inference
    )

    action_logits = out["action_logits"]   # [1, K, |A|]
    world_logits  = out["world_logits"]    # [1, N, V_wm]

    action_tokens = action_logits.argmax(dim=-1).squeeze(0).cpu()  # [K]
    world_tokens  = world_logits.argmax(dim=-1).squeeze(0).cpu()   # [N]

    primary_action_id = int(action_tokens[0].item())
    action_name = ACTION_NAMES.get(primary_action_id, "FORWARD")

    return {
        "action_tokens": action_tokens,
        "action_name":   action_name,
        "world_tokens":  world_tokens,
        "raw":           out,
    }
