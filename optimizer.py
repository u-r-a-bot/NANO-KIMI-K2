import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import math

class MuonClip(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        qk_clip_tau: float = 100.0,
        qk_clip_enabled: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            qk_clip_tau=qk_clip_tau,
            qk_clip_enabled=qk_clip_enabled
        )
        super().__init__(params, defaults)
        self.max_logits_history = []
    
    @torch.no_grad()
    def step(self, closure=None, max_logits: Optional[float] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if max_logits is not None and self.defaults['qk_clip_enabled']:
            self.max_logits_history.append(max_logits)
        
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            qk_clip_tau = group['qk_clip_tau']
            qk_clip_enabled = group['qk_clip_enabled']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                is_qk_weight = self._is_qk_weight(p, group)
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0
                
                buf = state['momentum_buffer']
                state['step'] += 1
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                buf.mul_(momentum).add_(grad)
                
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf
                
                if qk_clip_enabled and is_qk_weight and max_logits is not None:
                    if max_logits > qk_clip_tau:
                        gamma = min(1.0, qk_clip_tau / max_logits)
                        gamma_h = math.sqrt(gamma)
                        p.mul_(gamma_h)
                        update.mul_(gamma_h)
                
                p.add_(update, alpha=-lr)
        
        return loss
    
    def _is_qk_weight(self, param, group):
        if 'is_qk' in group:
            return group['is_qk']
        return False
    
    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

def get_muon_param_groups(model, lr=0.02, weight_decay=0.01):
    qk_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(x in name.lower() for x in ['wq', 'wk', 'q_proj', 'k_proj']):
            qk_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {
            'params': qk_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'is_qk': True
        },
        {
            'params': other_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'is_qk': False
        }
    ]
    
    return param_groups