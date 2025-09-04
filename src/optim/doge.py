import os
import pickle
import torch
import numpy as np

def get_model_grad_flat(model, tgt_params_ls=None):
    """Flatten gradients into a single vector."""
    full_grad_concat = None
    for p_name, p in model.named_parameters():
        if tgt_params_ls is not None and p_name not in tgt_params_ls:
            continue
        if "expert" in p_name:
            grad_norm = p.grad.norm().item() if p.grad is not None else None
            if p.grad is not None:
                flat_grad = p.grad.detach().flatten()
                if grad_norm != 0.0:
                    full_grad_concat = torch.cat([full_grad_concat, flat_grad])
                else:
                    full_grad_concat = flat_grad
    return full_grad_concat

def get_expert_grad_flat(model, expert_idx):
    expert = model.expert_routing.experts[expert_idx]
    flat_grad = []
    for _, p in expert.named_parameters():
        if p.grad is not None:
            flat_grad.append(p.grad.detach().flatten())
    return torch.cat(flat_grad) if flat_grad else None


def get_model_grad_dict(model, expert_idx):
    """Get gradients as dict {param_name: grad_tensor}."""
    expert = model.expert_routing.experts[expert_idx]
    return {p_name: p.grad for p_name, p in expert.named_parameters()}


def add_expert_grad_ls(model, domain_full_grad_dicts, dw):
    """Combine domain-specific grads into model.grad weighted by dw."""
    for expert in model.expert_routing.experts:
        for p_name, p in expert.named_parameters():
            grad_accum = None
            for idx, v in enumerate(dw):
                if domain_full_grad_dicts[idx] is not None and domain_full_grad_dicts[idx][p_name] is not None:
                    contrib = domain_full_grad_dicts[idx][p_name] * v
                    grad_accum = contrib if grad_accum is None else grad_accum + contrib
            if grad_accum is not None:
                if p.grad is None:
                    p.grad = grad_accum
                else:
                    p.grad += grad_accum


class DoGE:
    def __init__(self,
                 model,
                 num_experts,
                 args,
                 train_ids,
                 tgt_ids,
                 train_dw=None,
                 val_dw=None):

        self.model = model  # must be set later by doge.model = model
        self.args = args

        # domain setup
        self.domain_list = [f"MoE_{i}" for i in range(num_experts)]
        self.idx2domain = {i: dom for i, dom in enumerate(self.domain_list)}
        self.domain2idx = {dom: i for i, dom in self.idx2domain.items()}
        self.grad_dict_tracker = []
        self.weights_tracker = []

        self.train_ids = torch.tensor(train_ids, dtype=torch.long)
        self.tgt_ids   = torch.tensor(tgt_ids, dtype=torch.long)

        # domain weights
        if train_dw is None:
            self.train_dw = torch.ones(len(self.domain_list)).to("cuda") / len(self.train_ids)
        else:
            self.train_dw = torch.tensor(train_dw, dtype=torch.float).to("cuda")

        if val_dw is None:
            self.val_dw = torch.zeros(len(self.domain_list)).to("cuda")
            self.val_dw[self.tgt_ids] = 1.0 / len(self.tgt_ids)
        else:
            self.val_dw = torch.tensor(val_dw, dtype=torch.float).to("cuda")

        # DoGE hyperparams
        self.reweight_eps = 0.0
        self.mu = 0.005
        self.dw_min = 0.0
        self.dw_max = 5.00

        # bookkeeping
        self.flat_grad_mat = None   # allocated later when model is set
        self.iter_domain_losses = torch.zeros(len(self.domain_list))
        self.avg_dw = torch.zeros(len(self.domain_list)).to("cuda")
        self.dw_update_steps = 0

        self.last_dw_save_path = os.path.join(self.args["output_dir"], f'{self.args["run_id"]}_last_dw_config.pkl')
        self.avg_dw_save_path = os.path.join(self.args["output_dir"], f'{self.args["run_id"]}_avg_dw_config.pkl')

        # reload if checkpoint exists
        if os.path.exists(self.last_dw_save_path):
            with open(self.last_dw_save_path, 'rb') as trg:
                cur_domain_config_dict = pickle.load(trg)
                self.train_dw = cur_domain_config_dict['train_dw']
                self.dw_update_steps = cur_domain_config_dict['dw_update_steps']
            with open(self.avg_dw_save_path, 'rb') as trg:
                avg_domain_config_dict = pickle.load(trg)
                self.avg_dw = avg_domain_config_dict['train_dw'] * self.dw_update_steps
            print(f'Resumed DoGE from step {self.dw_update_steps}…')

    def _init_grad_buffers(self):
        """Allocate grad matrix once model is known."""
        if self.flat_grad_mat is None:
            num_params = sum(p.numel() for p in self.model.expert_routing.experts[0].parameters())
            self.flat_grad_mat = torch.zeros((len(self.domain_list), num_params), device=next(self.model.parameters()).device)

    def write_weights(self, cur_weights, avg_weights):
        cur_domain_config_dict = {'train_dw': cur_weights, 'dw_update_steps': self.dw_update_steps}
        avg_domain_config_dict = {'train_dw': avg_weights}
        with open(self.last_dw_save_path, 'wb') as trg:
            pickle.dump(cur_domain_config_dict, trg)
        with open(self.avg_dw_save_path, 'wb') as trg:
            pickle.dump(avg_domain_config_dict, trg)

    def __call__(self, pertoken_losses, token_masks, domain_ids, current_lr, reweight=False):
        """Update domain weights based on gradients (no backward here)."""
        assert self.model is not None, "doge.model = model must be set before use"
        self._init_grad_buffers()

        wandb_log_dict = {}

        full_grad_dicts = []
        all_domain_losses = []
        for domain_id in range(len(self.domain_list)):
            domain_mask = (domain_ids == domain_id)
            if domain_mask.sum() > 0:
                curr_domain_losses = pertoken_losses[token_masks.cpu() * domain_mask].mean()
                all_domain_losses.append(curr_domain_losses)
            else:
                all_domain_losses.append(None)

        # read gradients (already computed by outer backward)
        for domain_id, curr_domain_losses in enumerate(all_domain_losses):
            if curr_domain_losses is None:
                full_grad_dicts.append(None)
            else:
                self.iter_domain_losses[domain_id] = curr_domain_losses

                # get domain grad
                if self.args["max_grad_norm"] > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
                domain_flat_grad = get_expert_grad_flat(self.model, domain_id)
                domain_full_grad_dict = get_model_grad_dict(self.model, domain_id)
                self.flat_grad_mat[domain_id][:] = domain_flat_grad
                full_grad_dicts.append(domain_full_grad_dict)

        train_mat = self.flat_grad_mat[self.train_ids][:]
        tgt_mat = self.flat_grad_mat[self.tgt_ids][:]
        scores_mat = train_mat @ tgt_mat.T

        if set(self.train_ids.tolist()) == set(self.tgt_ids.tolist()):
            scores = current_lr * (scores_mat.sum(dim=-1) - scores_mat.diag())
        else:
            scores = current_lr * scores_mat.sum(dim=-1)

        avg_norm = train_mat.norm(dim=-1).mean()
        scores = scores / (avg_norm + 1e-6)
        scores = torch.clip(scores, min=current_lr * self.dw_min, max=current_lr * self.dw_max)

        dw_prev = self.train_dw
        log_dw_new = torch.log(dw_prev[self.train_ids]) + scores / self.mu
        dw_new = torch.softmax(log_dw_new, dim=-1)
        dw_new = (1 - self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new)
        self.train_dw[self.train_ids] = dw_new

        if not reweight:
            self.weights_tracker.append(self.train_dw.cpu().numpy())
            self.grad_dict_tracker.append({i: full_grad_dicts[i] for i in self.train_ids.tolist()})
        else:
            self.weights_tracker.append(self.train_dw.cpu().numpy())
            self.grad_dict_tracker.append({i: full_grad_dicts[i] for i in self.train_ids.tolist()})
            weights_stack = torch.tensor(np.array(self.weights_tracker))
            agg_dw_weights = weights_stack.mean(dim=0).to(device=self.avg_dw.device)

            self.avg_dw[self.train_ids] += agg_dw_weights
            self.dw_update_steps += 1

            avg_grad_dict = {}
            for i in self.train_ids.tolist():
                step_dicts = [d[i] for d in self.grad_dict_tracker if d[i] is not None]
                if not step_dicts:
                    avg_grad_dict[i] = None
                    continue
                keys = step_dicts[0].keys()
                avg_grad_dict[i] = {k: sum(sd[k] for sd in step_dicts) / len(step_dicts) for k in keys}

            add_expert_grad_ls(self.model, [avg_grad_dict[i] for i in self.train_ids.tolist()], dw=agg_dw_weights[self.train_ids])
            self.write_weights(cur_weights=agg_dw_weights, avg_weights=self.avg_dw / self.dw_update_steps)
            self.grad_dict_tracker.clear()
            self.weights_tracker.clear()

            grad_norm = self.flat_grad_mat.norm(dim=-1)
            for domain_idx in range(len(self.domain_list)):
                domain_name = self.idx2domain[domain_idx]
                if domain_idx in self.train_ids:
                    wandb_log_dict[f'score/{domain_name}'] = scores[domain_idx].item()
                elif domain_idx in self.tgt_ids:
                    wandb_log_dict[f'score/{domain_name}'] = 0.0
                wandb_log_dict[f'grad_norm/{domain_name}'] = max(grad_norm[domain_idx].item(), self.args["max_grad_norm"])
                wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
                wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
                wandb_log_dict[f'loss/{domain_name}'] = self.iter_domain_losses[domain_idx]

            return wandb_log_dict
