from collections import deque
from pathlib import Path
import torch
from .checkpoint import restore, check_tokenizer
from .actions import action_id, ACTION_NAMES
from .data import image_sequence

class LivePolicy:
    """Adapter for a future simulator integration. No Unity executable assumed.

    push_frame is called for every 10 Hz observation; decide at 2 Hz. Pass the
    LAST ACTUALLY EXECUTED action (after intervention), never the expert label
    for the decision being predicted. Reset explicitly between episodes.
    """
    def __init__(self, checkpoint, device="cpu"):
        from transformers import CLIPTokenizerFast
        self.model, self.metadata = restore(checkpoint, "policy")
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.cfg = self.model.cfg
        self.tokenizer = CLIPTokenizerFast.from_pretrained(check_tokenizer(self.metadata, checkpoint), local_files_only=True)
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.cfg.memory_frames*self.cfg.keyframe_stride+1)
        self.last_timestamp = None
        self.last_decision = None

    def push_frame(self, path, timestamp_s):
        if self.last_timestamp is not None and abs(timestamp_s-self.last_timestamp-self.cfg.frame_period_s) > .02*self.cfg.frame_period_s:
            raise ValueError("Irregular live frame timing; resample or reset explicitly")
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        self.frames.append(path)
        self.last_timestamp = timestamp_s

    @torch.no_grad()
    def decide(self, instruction, last_executed_action):
        if not self.frames: raise ValueError("No observation received")
        period = self.cfg.frame_period_s*self.cfg.keyframe_stride
        if self.last_decision is not None and self.last_timestamp-self.last_decision < period-1e-6:
            raise ValueError("Decision requested before the next control interval")
        action_id(last_executed_action)
        frames = list(self.frames)
        paths = [frames[max(0, len(frames)-1-(self.cfg.memory_frames-j)*self.cfg.keyframe_stride)]
                 for j in range(self.cfg.memory_frames+1)]
        images = image_sequence(paths, self.cfg.image_size)[None].to(self.device)
        enc = self.tokenizer(instruction, padding="max_length", truncation=True,
                             max_length=self.cfg.max_text_length, return_tensors="pt")
        logits = self.model(images, enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
                            torch.tensor([last_executed_action], dtype=torch.long, device=self.device))
        probs = logits.softmax(-1)[0].cpu().tolist()
        selected = int(logits.argmax(-1))
        self.last_decision = self.last_timestamp
        return {"action_id": selected, "action_name": ACTION_NAMES[selected], "probabilities": probs}
