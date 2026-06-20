# KL-FreshPER Design Note

## Goal

FreshPER uses age as a proxy for policy drift:

```text
p_i(t) = b_i(t) * exp(-(t - t_i) / tau)
```

where `b_i` is a base PER signal such as `|reward| + eps`.

KL-FreshPER keeps this cold-start behavior, but when a replay sample is actually
trained, it reuses the actor training forward pass to estimate the current
policy divergence from the stored behavior policy and writes back a KL-decayed
priority.

## Divergence Estimate

For a sampled trajectory, the replay buffer stores behavior log-probs:

```text
old_log_probs = log pi_mu(a_t | s_t)
```

The actor training step can return current log-probs:

```text
current_log_probs = log pi_theta(a_t | s_t)
```

Define:

```text
log_ratio_t = current_log_probs_t - old_log_probs_t
r_t = exp(log_ratio_t)
```

The reverse-KL direction used in the FreshPER ESS derivation is:

```text
D_KL(pi_theta || pi_mu) = E_mu[r_t * log r_t]
```

Using the zero-mean control variate `E_mu[r_t - 1] = 0`, the implementation
uses the non-negative estimator:

```text
d_i = mean_t [r_t * log r_t - r_t + 1]
```

computed over response tokens only.

## Priority

The cached write-back priority is:

```text
p_i = (|reward_i| + eps) * exp(-eta * d_i)
```

If `enable_age_decay: true`, the replay buffer still applies residual freshness
decay to the cached priority:

```text
p_i(t) = p_i * exp(-(t - t_obs_i) / tau)
```

For samples that have never been replayed, `kl_fresh` starts from reward-base
priority and therefore behaves like FreshPER cold start with age decay.

## Configuration

```yaml
replay:
  enabled: true
  priority_function: "kl_fresh"
  priority_exponent: 0.6
  enable_age_decay: true
  age_decay: 1000
  kl_fresh_eta: 1.0
  kl_fresh_log_ratio_clip: 10.0
```

## Notes

- This is not `priority = KL`. High KL means the sample is more off-policy, so
  it must be down-weighted.
- The method does not recompute full policy distributions. It reuses chosen-token
  log-probs from the replay actor training forward pass.
- No full-buffer actor forward is required. Only sampled replay rows get observed
  KL write-back; unsampled rows continue to decay by age.
