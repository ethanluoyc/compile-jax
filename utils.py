"""Utility functions."""

import numpy as np

import jax
import jax.numpy as jnp
from jax import nn, random

EPS = 1e-17
NEG_INF = -1e30


def cross_entropy(logits, labels, reduction='none'):
  sequence_length, batch_size = logits.shape[:2]
  targets = jax.nn.one_hot(labels, logits.shape[-1])
  return -jnp.sum(targets * jax.nn.log_softmax(logits, -1), -1)


def gumbel_sample(rng, shape):
  """Sample Gumbel noise."""
  uniform = random.uniform(rng, shape=shape)
  return -jnp.log(EPS - jnp.log(uniform + EPS))


def gumbel_softmax_sample(rng, logits, temp=1.):
  """Sample from the Gumbel softmax / concrete distribution."""
  gumbel_noise = gumbel_sample(rng, logits.shape)
  return nn.softmax((logits + gumbel_noise) / temp, axis=-1)


def gaussian_sample(rng, mu, log_var):
  """Sample from Gaussian distribution."""
  gaussian_noise = random.normal(rng, mu.shape)
  return mu + jnp.exp(log_var * 0.5) * gaussian_noise


def kl_gaussian(mu, log_var):
  """KL divergence between Gaussian posterior and standard normal prior."""
  return -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var), axis=1)


def kl_categorical_uniform(preds):
  """KL divergence between categorical distribution and uniform prior."""
  kl_div = preds * jnp.log(preds + EPS)  # Constant term omitted.
  return kl_div.sum(1)


def kl_categorical(preds, log_prior):
  """KL divergence between two categorical distributions."""
  kl_div = preds * (jnp.log(preds + EPS) - log_prior)
  return kl_div.sum(1)


def poisson_categorical_log_prior(length, rate):
  """Categorical prior populated with log probabilities of Poisson dist."""
  rate = jnp.array(rate, dtype=jnp.float32)
  values = jnp.expand_dims(jnp.arange(1, length + 1, dtype=jnp.float32), 0)
  log_prob_unnormalized = jax.lax.lgamma(
      jnp.log(rate) * values - rate - (values + 1))
  # TODO(tkipf): Length-sensitive normalization.
  return nn.log_softmax(log_prob_unnormalized, axis=1)  # Normalize.


def log_cumsum(probs, axis=1):
  """Calculate log of inclusive cumsum."""
  return jnp.log(jnp.cumsum(probs, axis=axis) + EPS)


def generate_toy_data(num_symbols=5, num_segments=3, max_segment_len=5):
  """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
  seq = []
  symbols = np.random.choice(
      np.arange(1, num_symbols + 1), num_segments, replace=False)
  for seg_id in range(num_segments):
    segment_len = np.random.choice(np.arange(1, max_segment_len))
    seq += [symbols[seg_id]] * segment_len
  seq += [0]
  return np.array(seq, dtype=jnp.int64)


def get_lstm_initial_state(batch_size, hidden_dim):
  """Get empty (zero) initial states for LSTM."""
  hidden_state = jnp.zeros((batch_size, hidden_dim))
  cell_state = jnp.zeros((batch_size, hidden_dim))
  return hidden_state, cell_state


def get_segment_probs(all_b_samples, all_masks, segment_id):
  """Get segment probabilities for a particular segment ID."""
  neg_cumsum = 1 - jnp.cumsum(all_b_samples[segment_id], axis=1)
  if segment_id > 0:
    return neg_cumsum * all_masks[segment_id - 1]
  else:
    return neg_cumsum


def get_losses(
    inputs,
    outputs,
    args,
    beta_b=.1,
    beta_z=.1,
    prior_rate=3.,
):
  """Get losses (NLL, KL divergences and neg. ELBO).

    Args:
        inputs: Padded input sequences.
        outputs: CompILE model output tuple.
        args: Argument dict from `ArgumentParser`.
        beta_b: Scaling factor for KL term of boundary variables (b).
        beta_z: Scaling factor for KL term of latents (z).
        prior_rate: Rate (lambda) for Poisson prior.
    """

  targets = inputs.reshape(-1)
  all_encs, all_recs, all_masks, all_b, all_z = outputs
  input_dim = args.num_symbols + 1

  nll = 0.
  kl_z = 0.
  for seg_id in range(args.num_segments):
    seg_prob = get_segment_probs(all_b['samples'], all_masks, seg_id)
    preds = all_recs[seg_id].reshape(-1, input_dim)
    seg_loss = cross_entropy(
        preds, targets, reduction='none').reshape(-1, inputs.shape[1])
    # print(seg_loss.shape, seg_prob.shape)

    # Ignore EOS token (last sequence element) in loss.
    nll += (seg_loss[:, :-1] * seg_prob[:, :-1]).sum(1).mean(0)

    # KL divergence on z.
    if args.latent_dist == 'gaussian':
      mu, log_var = jnp.split(all_z['logits'][seg_id], 2, axis=1)
      kl_z += kl_gaussian(mu, log_var).mean(0)
    elif args.latent_dist == 'concrete':
      kl_z += kl_categorical_uniform(
          nn.softmax(all_z['logits'][seg_id], axis=-1)).mean(0)
    else:
      raise ValueError('Invalid argument for `latent_dist`.')

  # KL divergence on b (first segment only, ignore first time step).
  # TODO(tkipf): Implement alternative prior on soft segment length.
  probs_b = nn.softmax(all_b['logits'][0], axis=-1)
  log_prior_b = poisson_categorical_log_prior(probs_b.shape[1], prior_rate)
  kl_b = args.num_segments * kl_categorical(probs_b[:, 1:],
                                            log_prior_b[:, 1:]).mean(0)

  loss = nll + beta_z * kl_z + beta_b * kl_b
  return loss, nll, kl_z, kl_b


def get_reconstruction_accuracy(inputs, outputs, args):
  """Calculate reconstruction accuracy (averaged over sequence length)."""

  all_encs, all_recs, all_masks, all_b, all_z = outputs

  batch_size = inputs.shape[0]

  rec_seq = []
  rec_acc = 0.
  for sample_idx in range(batch_size):
    prev_boundary_pos = 0
    rec_seq_parts = []
    for seg_id in range(args.num_segments):
      boundary_pos = jnp.argmax(all_b['samples'][seg_id], axis=-1)[sample_idx]
      if prev_boundary_pos > boundary_pos:
        boundary_pos = prev_boundary_pos
      seg_rec_seq = jnp.argmax(all_recs[seg_id], axis=-1)
      rec_seq_parts.append(seg_rec_seq[sample_idx,
                                       prev_boundary_pos:boundary_pos])
      prev_boundary_pos = boundary_pos
    rec_seq.append(jnp.concatenate(rec_seq_parts))
    cur_length = rec_seq[sample_idx].shape[0]
    matches = rec_seq[sample_idx] == inputs[sample_idx, :cur_length]
    rec_acc += matches.astype(jnp.float32).mean()
  rec_acc /= batch_size
  return rec_acc, rec_seq
