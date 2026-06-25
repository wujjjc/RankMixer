import torch
from dataclasses import dataclass, field


def _query_best_device():
    """查询空闲显存最多的GPU，不创建持久context"""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return torch.device('cuda:0')
    max_free = -1
    best_device = 0
    for i in range(num_gpus):
        try:
            with torch.cuda.device(i):  # 临时作用域，不残留context
                free, _ = torch.cuda.mem_get_info(i)
                if free > max_free:
                    max_free = free
                    best_device = i
        except RuntimeError:
            continue
    if max_free < 0:
        return torch.device('cpu')
    return torch.device(f'cuda:{best_device}')


class _LazyDevice:
    """延迟初始化描述符，首次访问 Config.device 时才查询GPU"""
    def __set_name__(self, owner, name):
        self._attr = f'_cached_{name}'

    def __get__(self, obj, objtype=None):
        cache_on = obj if obj is not None else objtype
        if not hasattr(cache_on, self._attr):
            setattr(cache_on, self._attr, _query_best_device())
        return getattr(cache_on, self._attr)


@dataclass
class Config:
    epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 0.001
    embedding_dim: int = 64
    end_time: int = 1494518400
    max_seq_len: int = 100
    device: torch.device = field(default=_LazyDevice(), init=False, repr=False)
    num_gender: int = 3
    num_age_level: int = 7
    num_pvalue_level: int = 4
    num_shopping_level: int = 4
    num_occupation: int = 2
    num_new_user_class_level: int = 5
    dropout: float = 0.1
    weight_decay: float = 1e-5
    eval_interval: int = 5
    token_dim: int = 128
    lamb: float = 1e-8
    warmup_steps: int = 100000
    grad_clip_norm: float = 1.0
