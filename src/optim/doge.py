import os
import pickle
import torch
import numpy as np

def get_model_grad_flat(model, tgt_params_ls=None):
    """Flatten gradients for a single expert into a vector."""
    grads = []
    for p_name, p in model.named_parameters():
        if tgt_params_ls is not None and p_name not in tgt_params_ls:
            continue
        if p.grad is not None and torch.norm(p.grad) > 0:
            flat_grad = p.grad.detach().flatten()
            grads.append(flat_grad)
    return torch.cat(grads) if grads else None

def get_model_grad_dict(model):
    """Get gradients for a single expert as dict {param_name: grad_tensor}."""
    return {p_name: p.grad.clone() for p_name, p in model.named_parameters() if p.grad is not None and torch.norm(p.grad) > 0}

def add_model_grad(model, domain_full_grad_dict, dw):
    for p_name, p in model.named_parameters():
        is_expert_param = 'expert' in p_name or 'experts' in p_name
        for idx, v in enumerate(dw):
            if domain_full_grad_dict[idx] is not None:
                if not is_expert_param:
                    if p.grad is None:
                        p.grad = domain_full_grad_dict[idx][p_name]*v
                    else:
                        p.grad += domain_full_grad_dict[idx][p_name]*v

                elif is_expert_param and p_name in domain_full_grad_dict[idx]:
                    p.grad = domain_full_grad_dict[idx][p_name]

class DoGE:
    def __init__(self,
                model,
                num_experts,
                args,
                train_ids,
                tgt_ids,
                train_dw=None,
                val_dw=None):

        self.model = model
        self.args = args
        self.num_experts = num_experts

        # domain setup
        self.domain_list = [f"MoE_{i}" for i in range(num_experts)]
        self.idx2domain = {i: dom for i, dom in enumerate(self.domain_list)}
        self.domain2idx = {dom: i for i, dom in self.idx2domain.items()}

        self.train_ids = torch.tensor(train_ids, dtype=torch.long)
        self.tgt_ids   = torch.tensor(tgt_ids, dtype=torch.long)

        # domain weights - one per expert
        if train_dw is None:
            self.train_dw = torch.ones(self.num_experts).to("cuda") / len(self.train_ids)
        else:
            self.train_dw = torch.tensor(train_dw, dtype=torch.float).to("cuda")

        if val_dw is None:
            self.val_dw = torch.zeros(self.num_experts).to("cuda")
            self.val_dw[self.tgt_ids] = 1.0 / len(self.tgt_ids)
        else:
            self.val_dw = torch.tensor(val_dw, dtype=torch.float).to("cuda")

        # DoGE hyperparams
        self.reweight_eps = 0.0
        self.mu = 0.05
        self.dw_min = 0.0
        self.dw_max = 5.00

        # bookkeeping - now we have separate gradient storage for each expert
        self.flat_grad_mat = None
        self.tgt_accumulation = None
        self.accumulation_steps = 0
        self.iter_domain_losses = dict() # torch.zeros(self.num_experts).to("cuda")
        self.avg_dw = torch.zeros(self.num_experts).to("cuda")
        self.dw_update_steps = 0

        self._init_grad_buffers()

        self.last_dw_save_path = os.path.join(self.args["output_dir"], f'{self.args["run_id"]}_last_dw_config.pkl')
        self.avg_dw_save_path = os.path.join(self.args["output_dir"], f'{self.args["run_id"]}_avg_dw_config.pkl')

        # reload if checkpoint exists
        if os.path.exists(self.last_dw_save_path):
            with open(self.last_dw_save_path, 'rb') as trg:
                cur_domain_config_dict = pickle.load(trg)
                self.train_dw = cur_domain_config_dict['train_dw'].clone()
                self.dw_update_steps = cur_domain_config_dict['dw_update_steps']
            with open(self.avg_dw_save_path, 'rb') as trg:
                avg_domain_config_dict = pickle.load(trg)
                self.avg_dw = avg_domain_config_dict['train_dw'] * self.dw_update_steps.clone()
            print(f'Resumed DoGE from step {self.dw_update_steps}…')

    def _init_grad_buffers(self):
        """Allocate grad matrix once model is known."""
        if self.flat_grad_mat is None:
            num_model_params = sum(p.numel() for p in self.model.parameters())
            num_expert_params = sum(p.numel() for p in self.model.expert_routing.experts[0].parameters())
            num_activated_params = num_model_params - (self.num_experts - 1) * num_expert_params
            self.flat_grad_mat = torch.zeros((self.num_experts, num_activated_params), device=next(self.model.parameters()).device)
            self.tgt_accumulation = torch.zeros((self.num_experts, num_activated_params), device=next(self.model.parameters()).device)

    def write_weights(self, cur_weights, avg_weights):
        cur_domain_config_dict = {'train_dw': cur_weights, 'dw_update_steps': self.dw_update_steps}
        avg_domain_config_dict = {'train_dw': avg_weights}
        with open(self.last_dw_save_path, 'wb') as trg:
            pickle.dump(cur_domain_config_dict, trg)
        with open(self.avg_dw_save_path, 'wb') as trg:
            pickle.dump(avg_domain_config_dict, trg)

    def __call__(self, pertoken_losses, domain_ids, current_lr, reweight=False):
        """Simplified DoGE for MoE where domain == expert."""
        wandb_log_dict = {}

        for domain_id in range(self.num_experts):
            domain_mask = (domain_ids == domain_id)
            if domain_mask.sum() > 0:
                # Compute loss and gradients for this domain
                curr_domain_loss = pertoken_losses[domain_mask].mean()
                should_retain = reweight or (domain_id < self.num_experts - 1)
                grads = torch.autograd.grad(curr_domain_loss, 
                                            self.model.parameters(), 
                                            retain_graph=should_retain,
                                            allow_unused=True)
                
                # Store the loss 
                if reweight:
                    self.iter_domain_losses[domain_id] = curr_domain_loss
                else:
                    self.iter_domain_losses[domain_id] = curr_domain_loss.item()

                # Get and store the gradients
                model_flat_grad = torch.cat([g.reshape(-1) for g in grads if g is not None and torch.norm(g) > 0])

                if domain_id in self.train_ids.tolist():
                    self.flat_grad_mat[domain_id][:] = model_flat_grad
                    self.tgt_accumulation[domain_id][:] += model_flat_grad
                
                if not reweight:
                    del curr_domain_loss
                del grads, model_flat_grad, domain_mask

        torch.cuda.empty_cache()
        self.accumulation_steps += 1
        
        # Reweighting logic
        if reweight and self.accumulation_steps > 0:
            train_grads = self.flat_grad_mat[self.train_ids]
            avg_tgt_grads = self.tgt_accumulation / self.accumulation_steps
            
            scores_mat = train_grads @ avg_tgt_grads.T

            if set(self.train_ids.tolist()) == set(self.tgt_ids.tolist()):
                scores = current_lr * (scores_mat.sum(dim=-1) - scores_mat.diag())
            else:
                scores = current_lr * scores_mat.sum(dim=-1)

            scores = scores / (train_grads.norm(dim=-1).mean() + 1e-6)
            scores = torch.clip(scores, min=self.dw_min, max=self.dw_max)
            
            dw_prev = self.train_dw.clone()
            log_dw_new = torch.log(dw_prev[self.train_ids]) + scores / self.mu
            dw_new = torch.softmax(log_dw_new, dim=-1)
            dw_new = (1 - self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new)
            
            self.train_dw[self.train_ids] = dw_new
            self.avg_dw += self.train_dw
            self.dw_update_steps += 1

            losses = torch.stack([self.iter_domain_losses[domain_id] for domain_id in range(self.num_experts)])
            weighted_losses = losses * self.train_dw
            total_loss = weighted_losses.sum()
            total_loss.backward()

            # Reset accumulator
            self.tgt_accumulation.zero_()
            self.accumulation_steps = 0
            
            for i, domain_idx in enumerate(self.train_ids.tolist()):
                domain_name = self.idx2domain[domain_idx]
                wandb_log_dict[f'expert_losses/{domain_name}'] = self.iter_domain_losses[domain_idx].item()
                wandb_log_dict[f'score/{domain_name}'] = scores[i].item()
                wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
                wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            return wandb_log_dict
            
        return None