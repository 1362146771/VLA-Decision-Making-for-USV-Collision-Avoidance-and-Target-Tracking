from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedContext:
    def __init__(self, device):
        self.size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0")) if self.size > 1 else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(device)
        self.owned = False
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            if self.size > 1:
                if not dist.is_nccl_available():
                    raise ValueError("CUDA DDP requires an NCCL-enabled PyTorch installation")
                self.device = torch.device("cuda", self.local_rank)
            elif self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        if self.size > 1:
            if dist.is_initialized():
                raise RuntimeError("Process group already initialized")
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=10))
            self.owned = True

    @property
    def primary(self):
        return self.rank == 0

    def close(self):
        if self.owned:
            dist.destroy_process_group()
            self.owned = False

    def primary_call(self, function):
        if self.size == 1:
            return function()
        payload = [None]
        if self.primary:
            try:
                payload[0] = (True, function())
            except Exception as exc:
                payload[0] = (False, f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(payload, src=0)
        ok, value = payload[0]
        if not ok:
            raise RuntimeError(value)
        return value

    def gather_objects(self, value):
        if self.size == 1:
            return [value]
        values = [None] * self.size
        dist.all_gather_object(values, value)
        return values

    def totals(self, *values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.size > 1:
            dist.all_reduce(tensor)
        return tensor.tolist()

    def check_finite(self, loss):
        valid = torch.isfinite(loss.detach()).all().to(dtype=torch.int32)
        if self.size > 1:
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise FloatingPointError("Non-finite training loss")


class ShardedEpochSampler(Sampler):
    def __init__(self, sampler, rank, size):
        self.sampler, self.rank, self.size = sampler, rank, size
        if size < 1 or not 0 <= rank < size:
            raise ValueError("rank/world_size")

    @property
    def epoch(self):
        return self.sampler.epoch

    @epoch.setter
    def epoch(self, value):
        self.sampler.epoch = value

    def __len__(self):
        return len(self.sampler) // self.size

    def __iter__(self):
        indices = list(self.sampler)
        return iter(indices[:len(self) * self.size][self.rank::self.size])


class GatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.rank = dist.get_rank()
        ctx.size = dist.get_world_size()
        parts = [torch.empty_like(value) for _ in range(ctx.size)]
        dist.all_gather(parts, value.contiguous())
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient)
        return gradient.chunk(ctx.size, dim=0)[ctx.rank].contiguous()


def contrastive_loss(image, text, logit_scale, distributed=False):
    if distributed:
        image = GatherWithGradient.apply(image)
        text = GatherWithGradient.apply(text)
    logits = logit_scale.exp() * image @ text.t()
    targets = torch.arange(len(image), device=image.device)
    loss = (torch.nn.functional.cross_entropy(logits, targets)
            + torch.nn.functional.cross_entropy(logits.t(), targets)) / 2
    return {"loss": loss, "logits": logits}
