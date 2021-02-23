import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn

import utils


class CompILE(hk.Module):
  """CompILE example implementation.

    Args:
        input_dim: Dictionary size of embeddings.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of latent variables (z).
        max_num_segments: Maximum number of segments to predict.
        temp_b: Gumbel softmax temperature for boundary variables (b).
        temp_z: Temperature for latents (z), only if latent_dist='concrete'.
        latent_dist: Whether to use Gaussian latents ('gaussian') or concrete /
            Gumbel softmax latents ('concrete').
    """

  def __init__(self,
               input_dim,
               hidden_dim,
               latent_dim,
               max_num_segments,
               temp_b=1.,
               temp_z=1.,
               latent_dist='gaussian',
               name='compile'):
    super().__init__(name=name)

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.max_num_segments = max_num_segments
    self.temp_b = temp_b
    self.temp_z = temp_z
    self.latent_dist = latent_dist

    self.embed = hk.Embed(input_dim, hidden_dim)
    self.lstm_cell = hk.LSTM(hidden_dim)

    # LSTM output heads.
    self.head_z_1 = hk.Linear(hidden_dim)  # Latents (z).

    if latent_dist == 'gaussian':
      self.head_z_2 = hk.Linear(latent_dim * 2)
    elif latent_dist == 'concrete':
      self.head_z_2 = hk.Linear(latent_dim)
    else:
      raise ValueError('Invalid argument for `latent_dist`.')

    self.head_b_1 = hk.Linear(hidden_dim)  # Boundaries (b).
    self.head_b_2 = hk.Linear(1)

    # Decoder MLP.
    self.decode_1 = hk.Linear(hidden_dim)
    self.decode_2 = hk.Linear(input_dim)

  def masked_encode(self, inputs, mask):
    """Run masked RNN encoder on input sequence."""
    # hidden = utils.get_lstm_initial_state(
    #     inputs.shape[0], self.hidden_dim)
    hidden = self.lstm_cell.initial_state(inputs.shape[0])
    outputs = []
    for step in range(inputs.shape[1]):
      _, hidden = self.lstm_cell(inputs[:, step], hidden)
      hidden = hk.LSTMState(mask[:, step, None] * hidden[0],
                            mask[:, step, None] * hidden[1])  # Apply mask.
      outputs.append(hidden[0])
    return jnp.stack(outputs, axis=1)

  def get_boundaries(self, encodings, segment_id, lengths, training):
    """Get boundaries (b) for a single segment in batch."""
    if segment_id == self.max_num_segments - 1:
      # Last boundary is always placed on last sequence element.
      logits_b = None
      # sample_b = jnp.zeros_like(encodings[:, :, 0]).scatter_(
      #     1, jnp.expand_dims(lengths, -1) - 1, 1)
      sample_b = jnp.zeros_like(encodings[:, :, 0])
      sample_b = jax.ops.index_update(
          sample_b, jax.ops.index[jnp.arange(len(lengths)), lengths - 1], 1)
    else:
      hidden = nn.relu(self.head_b_1(encodings))
      logits_b = jnp.squeeze(self.head_b_2(hidden), -1)
      # Mask out first position with large neg. value.
      neg_inf = jnp.ones((encodings.shape[0], 1)) * utils.NEG_INF
      # TODO(tkipf): Mask out padded positions with large neg. value.
      logits_b = jnp.concatenate([neg_inf, logits_b[:, 1:]], axis=1)
      if training:
        sample_b = utils.gumbel_softmax_sample(
            hk.next_rng_key(), logits_b, temp=self.temp_b)
      else:
        sample_b_idx = jnp.argmax(logits_b, axis=1)
        sample_b = nn.one_hot(sample_b_idx, logits_b.shape[1])

    return logits_b, sample_b

  def get_latents(self, encodings, probs_b, training):
    """Read out latents (z) form input encodings for a single segment."""
    readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
    readout = (encodings[:, :-1] * readout_mask).sum(1)
    hidden = nn.relu(self.head_z_1(readout))
    logits_z = self.head_z_2(hidden)

    # Gaussian latents.
    if self.latent_dist == 'gaussian':
      if training:
        mu, log_var = jnp.split(logits_z, 2, axis=1)
        sample_z = utils.gaussian_sample(hk.next_rng_key(), mu, log_var)
      else:
        sample_z = logits_z[:, :self.latent_dim]

    # Concrete / Gumbel softmax latents.
    elif self.latent_dist == 'concrete':
      if training:
        sample_z = utils.gumbel_softmax_sample(
            hk.next_rng_key(), logits_z, temp=self.temp_z)
      else:
        sample_z_idx = jnp.argmax(logits_z, axis=1)
        sample_z = utils.to_one_hot(sample_z_idx, logits_z.size(1))
    else:
      raise ValueError('Invalid argument for `latent_dist`.')

    return logits_z, sample_z

  def decode(self, sample_z, length):
    """Decode single time step from latents and repeat over full seq."""
    hidden = nn.relu(self.decode_1(sample_z))
    pred = self.decode_2(hidden)
    return jnp.tile(jnp.expand_dims(pred, 1), (1, length, 1))

  def get_next_masks(self, all_b_samples):
    """Get RNN hidden state masks for next segment."""
    if len(all_b_samples) < self.max_num_segments:
      # Product over cumsums (via log->sum->exp).
      log_cumsums = list(
          map(lambda x: utils.log_cumsum(x, axis=1), all_b_samples))
      mask = jnp.exp(sum(log_cumsums))
      return mask
    else:
      return None

  def __call__(self, inputs, lengths, training):

    # Embed inputs.
    embeddings = self.embed(inputs)

    # Create initial mask.
    mask = jnp.ones((inputs.shape[0], inputs.shape[1]))

    all_b = {'logits': [], 'samples': []}
    all_z = {'logits': [], 'samples': []}
    all_encs = []
    all_recs = []
    all_masks = []
    for seg_id in range(self.max_num_segments):

      # Get masked LSTM encodings of inputs.
      encodings = self.masked_encode(embeddings, mask)
      all_encs.append(encodings)

      # Get boundaries (b) for current segment.
      logits_b, sample_b = self.get_boundaries(encodings, seg_id, lengths,
                                               training)
      all_b['logits'].append(logits_b)
      all_b['samples'].append(sample_b)

      # Get latents (z) for current segment.
      logits_z, sample_z = self.get_latents(encodings, sample_b, training)
      all_z['logits'].append(logits_z)
      all_z['samples'].append(sample_z)

      # Get masks for next segment.
      mask = self.get_next_masks(all_b['samples'])
      all_masks.append(mask)

      # Decode current segment from latents (z).
      reconstructions = self.decode(sample_z, length=inputs.shape[1])
      all_recs.append(reconstructions)

    return all_encs, all_recs, all_masks, all_b, all_z
