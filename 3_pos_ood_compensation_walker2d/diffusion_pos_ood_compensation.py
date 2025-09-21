import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import sys
sys.path.append(("../"))
from torch.optim.lr_scheduler import CosineAnnealingLR
from agent.models import TanhGaussianPolicy, Critic

D4RL_SUPPRESS_IMPORT_ERROR=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Diffusion_pos_ood_compensation(object):
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            max_action, 
            replay_buffer, 
            behavior_model, 
            state_distribution,
            diffusion_model,
            dynamics_model,
            Q_min, 
            state_n_levels,
            action_n_levels,
            state_threshold, 
            action_threshold,
            beta,
            lam,
            omega,
            expectile,
            n_action_samples,
            comp_degree,
            discount=0.99, 
            tau=0.005, 
            policy_freq=2,
            target_update_freq=2, 
            schedule=True
        ):

        self.actor = TanhGaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().detach()

        self.replay_buffer = replay_buffer
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.target_update_freq = target_update_freq
        self.behavior_model = behavior_model
        self.state_distribution = state_distribution
        self.diffusion_model = diffusion_model
        self.dynamics_model = dynamics_model
        self.Q_min = Q_min
        self.state_n_levels = state_n_levels
        self.action_n_levels = action_n_levels
        self.state_threshold = state_threshold
        self.action_threshold = action_threshold
        self.beta = beta
        self.lam = lam
        self.omega = omega
        self.expectile = expectile
        self.n_action_samples = n_action_samples
        self.comp_degree = comp_degree
        self.schedule = schedule
        if schedule:
            self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(int(1e6) / policy_freq))
            self.critic_lr_schedule = CosineAnnealingLR(self.critic_optimizer, int(int(1e6) / policy_freq))

        # debug / safety settings
        self.debug_fail_fast = True   # 如果为 True，检测到 NaN/Inf 会抛错；设为 False 会尝试跳过该 step
        self.enable_autograd_anomaly = False  # 在需要时设为 True 可启用 torch.autograd.set_detect_anomaly(True)
        self.nan_check_every_backward = True  # 在每次 backward 后检查 gradients

        self.total_it = 0


    def compute_state_error(self, states, batch_size=256):
        recon_errors = []
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i + batch_size]
                batch_errors = torch.zeros(len(batch_states), device=states.device)

                for _ in range(self.state_n_levels):
                    t = self.diffusion_model.make_sample_density()(shape=(len(batch_states),), device=states.device)
                    noise = torch.randn_like(batch_states)

                    t_expanded = t.view(-1, *([1] * (batch_states.ndim - 1)))
                    noisy_states = batch_states + noise * t_expanded

                    c_skip, c_out, c_in = [
                        x.view(-1, *([1] * (batch_states.ndim - 1))) 
                        for x in self.diffusion_model.get_diffusion_scalings(t)
                    ]

                    model_input = noisy_states * c_in
                    model_output = self.state_distribution(model_input, None, torch.log(t) / 4)
                    denoised_states = c_skip * noisy_states + c_out * model_output

                    error = torch.norm(denoised_states - batch_states, dim=1)
                    batch_errors += error

                batch_errors /= self.state_n_levels
                recon_errors.append(batch_errors)

        return torch.cat(recon_errors)


    def compute_action_error(self, actions, states, batch_size=256):
        recon_errors = []
        with torch.no_grad():
            for i in range(0, len(actions), batch_size):
                batch_actions = actions[i:i + batch_size]
                batch_states = states[i:i + batch_size]
                batch_errors = torch.zeros(len(batch_actions), device=actions.device)

                for _ in range(self.action_n_levels):
                    t = self.diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                    noise = torch.randn_like(batch_actions)

                    t_expanded = t.view(-1, *([1] * (batch_actions.ndim - 1)))
                    noisy_actions = batch_actions + noise * t_expanded

                    c_skip, c_out, c_in = [
                        x.view(-1, *([1] * (batch_actions.ndim - 1))) 
                        for x in self.diffusion_model.get_diffusion_scalings(t)
                    ]

                    model_input = noisy_actions * c_in
                    model_output = self.behavior_model(model_input, batch_states, torch.log(t) / 4)
                    denoised_actions = c_skip * noisy_actions + c_out * model_output

                    error = torch.norm(denoised_actions - batch_actions, dim=1)
                    batch_errors += error

                batch_errors /= self.action_n_levels
                recon_errors.append(batch_errors)

        return torch.cat(recon_errors)


    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state)[0].cpu().data.numpy().flatten()
            self.actor.train()
            return action
        

    def select_best_id_action(self, state, max_resample=3):
        """
        稳健地从 diffusion 生成候选动作并选择 Q 最大者。
        - 会对采样结果做 NaN/Inf 检查并重采样（最多 max_resample 次）。
        - 在无法得到任何有效候选时回退到 actor(state)（优先）或 uniform。
        返回: best_id_actions (B, action_dim), best_q_values (B, 1)
        """
        with torch.no_grad():
            B = state.size(0)
            device = state.device
            n = self.n_action_samples
            action_dim = self.action_dim

            # 1) 初次采样（shape: [B, n, action_dim]）
            all_actions = self.diffusion_model.sample(
                model=self.behavior_model,
                cond=state,
                n_action_samples=n
            ).to(device)  # 确保在同一 device

            # 2) 如果含 NaN/Inf，尝试重采样（有限次）
            resample_attempts = 0
            while (not torch.isfinite(all_actions).all()) and (resample_attempts < max_resample):
                resample_attempts += 1
                try:
                    all_actions = self.diffusion_model.sample(
                        model=self.behavior_model,
                        cond=state,
                        n_action_samples=n
                    ).to(device)
                except Exception:
                    break

            # 3) 最终把候选动作先 clamp 到合法范围（避免极端值）
            all_actions = torch.clamp(all_actions, -self.max_action, self.max_action)

            # log stats for diagnostics
            nan_count = int((~torch.isfinite(all_actions)).sum().item())
            if nan_count > 0:
                print(f"[select_best_id_action] After resample attempts={resample_attempts}, NaN in all_actions: {nan_count}")
                try:
                    wandb.log({"debug/best_id_all_actions_nan_count": nan_count, "debug/best_id_resample_attempts": resample_attempts}, step=self.total_it)
                except Exception:
                    pass

            # 4) 批量评估 Q 值（按原有实现，把 states expand 后扁平）
            states_expanded = state.unsqueeze(1).expand(-1, n, -1)  # [B, n, state_dim]
            flat_states = states_expanded.reshape(-1, state.shape[-1])  # [B*n, state_dim]
            flat_actions = all_actions.reshape(-1, action_dim)  # [B*n, action_dim]

            # 评估 Q（返回 q shape: [B*n, 1]）
            q1, q2, q3, q4 = self.critic(flat_states, flat_actions)
            q_values_flat = torch.min(torch.min(q1, q2), torch.min(q3, q4)).view(B, n)  # [B, n]

            # 5) Mask 非 finite 的 q_values（包括那些因为 all_actions 出现 NaN 导致的 q NaN）
            finite_mask = torch.isfinite(q_values_flat)  # True 表示该候选有效
            valid_per_sample = finite_mask.any(dim=1)    # [B], 表示每个样本是否至少有一个有效候选

            # 将不可选项设为 -inf（这样 argmax 不会选到它们）
            q_values_masked = q_values_flat.clone()
            q_values_masked[~finite_mask] = -float("inf")

            # 6) 选索引
            best_indices = torch.argmax(q_values_masked, dim=1)  # [B]
            best_actions = torch.zeros((B, action_dim), device=device)
            best_qs = torch.zeros((B, 1), device=device)

            # 对有有效候选的样本取出对应 action / q
            if valid_per_sample.any():
                good_idx = torch.where(valid_per_sample)[0]
                best_actions[good_idx] = all_actions[good_idx, best_indices[good_idx], :]
                best_qs[good_idx] = q_values_flat[good_idx, best_indices[good_idx]].unsqueeze(-1)

            # 7) 对于没有任何有效候选的样本，按优先级回退：actor(state) -> behavior_model sample -> uniform
            fallback_idx = torch.where(~valid_per_sample)[0]
            if len(fallback_idx) > 0:
                # 优先使用当前 actor 输出（deterministic）
                actor_actions = self.actor(state[fallback_idx])[0]  # [K, action_dim]
                # 再次 guard：如果 actor 也含 NaN/Inf，则尝试从 diffusion 再采样一次作为备选，或用 uniform
                actor_finite = torch.isfinite(actor_actions).all(dim=1)
                if actor_finite.all():
                    best_actions[fallback_idx] = actor_actions
                    # 计算 critic q for these fallback actions
                    q1_f, q2_f, q3_f, q4_f = self.critic(state[fallback_idx], actor_actions)
                    best_qs[fallback_idx] = torch.min(torch.min(q1_f, q2_f), torch.min(q3_f, q4_f)).unsqueeze(-1)
                else:
                    # actor 也坏：尝试用 behavior_model 一次采样
                    try:
                        fallback_actions = self.diffusion_model.sample(model=self.behavior_model, cond=state[fallback_idx], n_action_samples=1).squeeze(1).to(device)
                        fallback_actions = torch.clamp(fallback_actions, -self.max_action, self.max_action)
                        if torch.isfinite(fallback_actions).all():
                            best_actions[fallback_idx] = fallback_actions
                            q1_f, q2_f, q3_f, q4_f = self.critic(state[fallback_idx], fallback_actions)
                            best_qs[fallback_idx] = torch.min(torch.min(q1_f, q2_f), torch.min(q3_f, q4_f)).unsqueeze(-1)
                        else:
                            # 最后退化方案：uniform random in [-max_action, max_action]
                            unif = torch.distributions.uniform.Uniform(-self.max_action, self.max_action)
                            rnd_actions = unif.sample((len(fallback_idx), action_dim)).to(device)
                            best_actions[fallback_idx] = rnd_actions
                            q1_f, q2_f, q3_f, q4_f = self.critic(state[fallback_idx], rnd_actions)
                            best_qs[fallback_idx] = torch.min(torch.min(q1_f, q2_f), torch.min(q3_f, q4_f)).unsqueeze(-1)
                    except Exception:
                        # 任何异常退回到 uniform
                        unif = torch.distributions.uniform.Uniform(-self.max_action, self.max_action)
                        rnd_actions = unif.sample((len(fallback_idx), action_dim)).to(device)
                        best_actions[fallback_idx] = rnd_actions
                        q1_f, q2_f, q3_f, q4_f = self.critic(state[fallback_idx], rnd_actions)
                        best_qs[fallback_idx] = torch.min(torch.min(q1_f, q2_f), torch.min(q3_f, q4_f)).unsqueeze(-1)

            # 8) 记录诊断量
            try:
                wandb.log({
                    "debug/best_id_nan_count": nan_count,
                    "debug/best_id_valid_ratio": float(valid_per_sample.float().mean().item()),
                    "debug/best_id_resample_attempts": resample_attempts,
                    "debug/best_id_fallback_count": int((~valid_per_sample).sum().item())
                }, step=self.total_it)
            except Exception:
                pass

            return best_actions, best_qs


    def alpha_loss(self, state):
        with torch.no_grad():
            _, log_prob = self.actor(state, need_log_prob=True)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        return alpha_loss
    

    def actor_loss(self, state):
        pi, pi_log_prob = self.actor(state, need_log_prob=True)
        pi_Q1, pi_Q2, pi_Q3, pi_Q4 = self.critic(state, pi)
        pi_Q = torch.cat([pi_Q1, pi_Q2, pi_Q3, pi_Q4], dim=1)
        pi_Q, _ = torch.min(pi_Q, dim=1)
        if pi_Q.mean().item() > 5e4:
            exit(0)
        actor_loss = (self.alpha * pi_log_prob - pi_Q).mean()
        return actor_loss
    

    def critic_loss(self, state, action, reward, next_state, not_done):
        def expectile_loss(diff, expectile):
            weight = torch.where(diff > 0, expectile, (1 - expectile))
            return weight * (diff ** 2)

        # helper for robust checks
        def tensor_stats(tensor):
            return {
                "nan": int(torch.isnan(tensor).sum().item()),
                "inf": int(torch.isinf(tensor).sum().item()),
                "mean": float(tensor.mean().detach().cpu().item()) if tensor.numel() > 0 else None,
                "min": float(tensor.min().detach().cpu().item()) if tensor.numel() > 0 else None,
                "max": float(tensor.max().detach().cpu().item()) if tensor.numel() > 0 else None,
                "abs_max": float(tensor.abs().max().detach().cpu().item()) if tensor.numel() > 0 else None
            }

        def check_and_handle(name, tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                stats = tensor_stats(tensor)
                msg = f"NUMERIC ISSUE: {name} contains NaN/Inf -> stats: {stats}"
                print(msg)
                try:
                    wandb.log({f"nan_check/{name}": stats}, step=self.total_it)
                except Exception:
                    pass
                if self.debug_fail_fast:
                    raise RuntimeError(msg)
                else:
                    return False
            return True

        # optional autograd anomaly detection
        if self.enable_autograd_anomaly:
            torch.autograd.set_detect_anomaly(True)

        with torch.no_grad():
            next_action, next_action_log_prob = self.actor_target(next_state, need_log_prob=True)

            next_Q1, next_Q2, next_Q3, next_Q4 = self.critic_target(next_state, next_action)
            next_Q = torch.min(torch.min(next_Q1, next_Q2), torch.min(next_Q3, next_Q4))
            next_Q = next_Q - self.alpha * next_action_log_prob
            target_Q = reward + not_done * self.discount * next_Q

            next_V1, next_V2 = self.critic_target.v(next_state)
            next_V = torch.min(next_V1, next_V2)
            target_V = reward + not_done * self.discount * next_V

        # basic checks on targets
        check_and_handle("target_Q", target_Q)
        check_and_handle("target_V", target_V)

        current_Q1, current_Q2, current_Q3, current_Q4 = self.critic(state, action)
        # check critic outputs
        check_and_handle("current_Q1", current_Q1)
        check_and_handle("current_Q2", current_Q2)
        check_and_handle("current_Q3", current_Q3)
        check_and_handle("current_Q4", current_Q4)

        current_Q = torch.cat([current_Q1, current_Q2, current_Q3, current_Q4], dim=1)

        current_V1, current_V2 = self.critic.v(state)
        q = torch.min(torch.min(current_Q1, current_Q2), torch.min(current_Q3, current_Q4))
        value_loss = expectile_loss(target_V - current_V1, self.expectile).mean() + \
                     expectile_loss(target_V - current_V2, self.expectile).mean()

        # Bellman loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
                      F.mse_loss(current_Q3, target_Q) + F.mse_loss(current_Q4, target_Q) + value_loss

        # action reconstruction error
        pi, _ = self.actor(state)
        # quick check actor output
        check_and_handle("pi", pi)

        pi_error = self.compute_action_error(pi, state)
        check_and_handle("pi_error", pi_error)

        # Predict next state s0
        pred_next_state = self.dynamics_model(state, pi)[..., :-1]  # [B, state_dim]
        check_and_handle("pred_next_state", pred_next_state)

        value_s_pi = self.critic.v_min(pred_next_state)
        # check v_min output
        check_and_handle("value_s_pi", value_s_pi)

        best_id_action, best_id_q = self.select_best_id_action(state)  # [B, action_dim], [B]
        # check outputs of select_best_id_action
        check_and_handle("best_id_action", best_id_action)
        check_and_handle("best_id_q", best_id_q)

        best_id_q = best_id_q.unsqueeze(1)  # [B, 1]
        best_id_next_state = self.dynamics_model(state, best_id_action)[..., :-1]  # [B, state_dim]
        check_and_handle("best_id_next_state", best_id_next_state)

        value_s_in = self.critic.v_min(best_id_next_state)
        check_and_handle("value_s_in", value_s_in)

        pred_next_state_error = self.compute_state_error(pred_next_state)
        check_and_handle("pred_next_state_error", pred_next_state_error)

        # NaN warnings collected earlier
        if torch.isnan(pi_error).any() or torch.isnan(pred_next_state_error).any():
            print("Warning: Reconstruction error contains NaN values!")
            if self.debug_fail_fast:
                raise RuntimeError("Reconstruction error contains NaN")

        ood_action_mask = (pi_error > self.action_threshold)
        ood_next_state_mask = (pred_next_state_error > self.state_threshold)
        negative_value_mask = (value_s_pi < value_s_in * self.comp_degree).squeeze(-1)
        positive_value_mask = (value_s_pi >= value_s_in * self.comp_degree).squeeze(-1)
        negative_ood_action_mask = (ood_action_mask & (ood_next_state_mask | negative_value_mask)).float().unsqueeze(1)
        positive_ood_action_mask = (ood_action_mask & (~ood_next_state_mask) & positive_value_mask).float().unsqueeze(1)

        pi_Q1, pi_Q2, pi_Q3, pi_Q4 = self.critic(state, pi)
        pi_Q = torch.cat([pi_Q1, pi_Q2, pi_Q3, pi_Q4], dim=1)
        check_and_handle("pi_Q", pi_Q)

        qmin = (self.Q_min * torch.ones_like(pi_Q)).detach()
        check_and_handle("qmin", qmin)

        reg_loss = self.beta * (((pi_Q - qmin) ** 2) * negative_ood_action_mask).mean()
        check_and_handle("reg_loss", reg_loss)

        value_diff = (value_s_pi - value_s_in).clamp(min=0.0)

        # form q_comp_target and add checks: detach + clamp
        q_comp_target = (best_id_q + value_diff).detach()
        # clamp to reasonable range to avoid extreme targets
        q_comp_target = torch.clamp(q_comp_target, min=-1e6, max=1e6)
        q_comp_target = self.omega * q_comp_target
        check_and_handle("q_comp_target", q_comp_target)

        vc_loss = self.lam * (((pi_Q - q_comp_target) ** 2) * positive_ood_action_mask).mean()
        check_and_handle("vc_loss", vc_loss)

        # final losses
        critic_loss = critic_loss + reg_loss + vc_loss

        # final check before returning
        check_and_handle("critic_loss", critic_loss)
        check_and_handle("reg_loss_final", reg_loss)
        check_and_handle("vc_loss_final", vc_loss)

        return critic_loss, reg_loss, vc_loss, current_Q, qmin, positive_ood_action_mask, negative_ood_action_mask, ood_action_mask, best_id_q, value_s_pi, value_s_in, pi_Q, pi_error, pred_next_state_error, positive_ood_action_mask, negative_ood_action_mask, q_comp_target


    def train(self, batch_size=256):
        self.total_it += 1

        # optional autograd anomaly enable
        if self.enable_autograd_anomaly:
            torch.autograd.set_detect_anomaly(True)

        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # basic input checks
        if torch.isnan(state).any() or torch.isinf(state).any() or torch.isnan(action).any() or torch.isinf(action).any():
            print("Warning: Input data contains NaN/Inf values!")
            print("state NaN/Inf:", int(torch.isnan(state).sum().item()), int(torch.isinf(state).sum().item()))
            print("action NaN/Inf:", int(torch.isnan(action).sum().item()), int(torch.isinf(action).sum().item()))
            # log to wandb
            try:
                wandb.log({
                    "nan_check/state_nan": int(torch.isnan(state).sum().item()),
                    "nan_check/state_inf": int(torch.isinf(state).sum().item()),
                    "nan_check/action_nan": int(torch.isnan(action).sum().item()),
                    "nan_check/action_inf": int(torch.isinf(action).sum().item()),
                }, step=self.total_it)
            except Exception:
                pass
            if self.debug_fail_fast:
                raise RuntimeError("ReplayBuffer returned NaN/Inf")
            else:
                return

        # Alpha update
        alpha_loss = self.alpha_loss(state)
        if not torch.isfinite(alpha_loss):
            print("alpha_loss not finite:", alpha_loss)
            raise RuntimeError("alpha_loss not finite")
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # optional grad check / clip
        if self.nan_check_every_backward:
            for name, p in self.alpha_optimizer.param_groups[0]['params'][0].named_parameters() if False else []:
                pass
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        if self.total_it % self.policy_freq == 0:
            actor_loss = self.actor_loss(state)
            if not torch.isfinite(actor_loss):
                print("actor_loss not finite:", actor_loss)
                raise RuntimeError("actor_loss not finite")
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)
            # gradient check
            if self.nan_check_every_backward:
                for p in self.actor.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print("NaN/Inf in actor grad detected")
                        if self.debug_fail_fast:
                            raise RuntimeError("NaN/Inf in actor gradients")
            actor_grad_norms = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0, norm_type=2)
            self.actor_optimizer.step()
            if self.schedule:
                try:
                    self.actor_lr_schedule.step()
                except Exception:
                    pass

        # Critic update
        critic_loss_tuple = self.critic_loss(state, action, reward, next_state, not_done)
        # if critic_loss raised earlier due to debug_fail_fast, it won't get here
        critic_loss, reg_loss, vc_loss, current_Q, qmin, positive_ood_action_mask, negative_ood_action_mask, ood_action_mask, best_id_q, value_s_pi, value_s_in, pi_Q, pi_error, pred_next_state_error, positive_ood_action_mask, negative_ood_action_mask, q_comp_target = critic_loss_tuple

        if not torch.isfinite(critic_loss):
            print("critic_loss not finite:", critic_loss)
            # log a bit more info then raise
            try:
                wandb.log({"debug/critic_loss_not_finite": 1}, step=self.total_it)
            except Exception:
                pass
            raise RuntimeError("critic_loss not finite")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient gate after backward
        if self.nan_check_every_backward:
            # check critic grads
            for name, p in self.critic.named_parameters():
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"NaN/Inf found in critic grad: {name}")
                    # print grad stats
                    g = p.grad
                    print("grad stats:", float(g.mean().detach().cpu().item()), float(g.abs().max().detach().cpu().item()))
                    try:
                        wandb.log({"debug/grad_nan_found": 1}, step=self.total_it)
                    except Exception:
                        pass
                    if self.debug_fail_fast:
                        raise RuntimeError(f"NaN/Inf in critic gradient {name}")
        critic_grad_norms = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0, norm_type=2)
        self.critic_optimizer.step()
        if self.schedule:
            try:
                self.critic_lr_schedule.step()
            except Exception:
                pass

        pos_mask = positive_ood_action_mask.float()
        neg_mask = negative_ood_action_mask.float()
        ood_mask = ood_action_mask.float()

        ood_count = ood_mask.sum().item()
        if ood_count > 0:
            pos_ratio = (pos_mask.sum() / ood_count).item()
            neg_ratio = (neg_mask.sum() / ood_count).item()
        else:
            pos_ratio, neg_ratio = 0.0, 0.0

        # Target networks update
        if self.total_it % self.target_update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # periodic logging (kept as in your code)
        if self.total_it % 10000 == 0:
            with torch.no_grad():
                try:
                    pi = self.actor(state)[0]
                except Exception as e:
                    print("Error when calling actor for logging:", e)
                    if self.debug_fail_fast:
                        raise
                    else:
                        pi = torch.zeros_like(action)

                unif = torch.distributions.uniform.Uniform(-1, 1).sample((batch_size, self.action_dim)).to(device)
                anoise1 = (action + torch.randn_like(action) * 0.1).clamp(-self.max_action, self.max_action)
                anoise5 = (action + torch.randn_like(action) * 0.5).clamp(-self.max_action, self.max_action)
                pinoise1 = (pi + torch.randn_like(action) * 0.1).clamp(-self.max_action, self.max_action)
                pinoise5 = (pi + torch.randn_like(action) * 0.5).clamp(-self.max_action, self.max_action)
                Q_pi1, Q_pi2, Q_pi3, Q_pi4 = self.critic(state, pi)
                Q_pi = torch.cat([Q_pi1, Q_pi2, Q_pi3, Q_pi4],dim=1)
                Q_unif1, Q_unif2, Q_unif3, Q_unif4 = self.critic(state, unif)
                Q_unif = torch.cat([Q_unif1, Q_unif2, Q_unif3, Q_unif4],dim=1)
                Q_anoise1_1, Q_anoise1_2, Q_anoise1_3, Q_anoise1_4 = self.critic(state, anoise1)
                Q_anoise1 = torch.cat([Q_anoise1_1, Q_anoise1_2, Q_anoise1_3, Q_anoise1_4],dim=1)
                Q_anoise5_1, Q_anoise5_2, Q_anoise5_3, Q_anoise5_4 = self.critic(state, anoise5)
                Q_anoise5 = torch.cat([Q_anoise5_1, Q_anoise5_2, Q_anoise5_3, Q_anoise5_4],dim=1)
                Q_pinoise1_1, Q_pinoise1_2, Q_pinoise1_3, Q_pinoise1_4 = self.critic(state, pinoise1)
                Q_pinoise1 = torch.cat([Q_pinoise1_1, Q_pinoise1_2, Q_pinoise1_3, Q_pinoise1_4],dim=1)
                Q_pinoise5_1, Q_pinoise5_2, Q_pinoise5_3, Q_pinoise5_4 = self.critic(state, pinoise5)
                Q_pinoise5 = torch.cat([Q_pinoise5_1, Q_pinoise5_2, Q_pinoise5_3, Q_pinoise5_4],dim=1)

                wandb.log({"train/critic_loss": critic_loss.item(),
                            "train/reg_loss": reg_loss.item(),
                            "train/vc_loss": vc_loss.item(),
                            "train/actor_loss": actor_loss.item() if 'actor_loss' in locals() else 0.0,
                            'Q/Qmin': qmin.mean().item(),
                            'Q/pi': Q_pi.mean().item(),
                            'Q/a': current_Q.mean().item(),
                            'Q/unif': Q_unif.mean().item(),
                            'Q/anoise0.1': Q_anoise1.mean().item(),
                            'Q/anoise0.5': Q_anoise5.mean().item(),
                            'Q/pinoise0.1': Q_pinoise1.mean().item(),
                            'Q/pinoise0.5': Q_pinoise5.mean().item(),
                            "ood/pos_ratio": pos_ratio,
                            "ood/neg_ratio": neg_ratio,
                            "ood/count_total": ood_count
                            }, step=self.total_it)

                wandb.log({
                    "debug/best_id_q_mean": best_id_q.mean().item(),
                    "debug/best_id_q_max": best_id_q.max().item(),
                    "debug/value_s_pi_mean": value_s_pi.mean().item(),
                    "debug/value_s_in_mean": value_s_in.mean().item(),
                    "debug/pi_Q_mean": pi_Q.mean().item(),
                    "debug/pi_error_mean": pi_error.mean().item(),
                    "debug/pred_next_state_error_mean": pred_next_state_error.mean().item(),
                    "debug/pos_ood_count": positive_ood_action_mask.sum().item(),
                    "debug/neg_ood_count": negative_ood_action_mask.sum().item(),
                    "debug/q_comp_target_mean": q_comp_target.mean().item(),
                    "debug/q_comp_target_max": q_comp_target.max().item(),
                    "debug/actor_grad_norms": actor_grad_norms if 'actor_grad_norms' in locals() else 0.0,
                    "debug/critic_grad_norms": critic_grad_norms if 'critic_grad_norms' in locals() else 0.0
                }, step=self.total_it)