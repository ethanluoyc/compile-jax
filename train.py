import argparse

import haiku as hk
import optax
import numpy as np

import jax
import jax.numpy as jnp
import modules
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--iterations',
    type=int,
    default=100,
    help='Number of training iterations.')
parser.add_argument(
    '--learning-rate', type=float, default=1e-2, help='Learning rate.')
parser.add_argument(
    '--hidden-dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument(
    '--latent-dim',
    type=int,
    default=32,
    help='Dimensionality of latent variables.')
parser.add_argument(
    '--latent-dist',
    type=str,
    default='gaussian',
    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument(
    '--batch-size',
    type=int,
    default=512,
    help='Mini-batch size (for averaging gradients).')

parser.add_argument(
    '--num-symbols',
    type=int,
    default=5,
    help='Number of distinct symbols in data generation.')
parser.add_argument(
    '--num-segments',
    type=int,
    default=3,
    help='Number of segments in data generation.')

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='Disable CUDA training.')
parser.add_argument(
    '--log-interval', type=int, default=5, help='Logging interval.')

args = parser.parse_args()


def build_model():
  model = modules.CompILE(
      input_dim=args.num_symbols + 1,  # +1 for EOS/Padding symbol.
      hidden_dim=args.hidden_dim,
      latent_dim=args.latent_dim,
      max_num_segments=args.num_segments,
      latent_dist=args.latent_dist)
  return model


def pad_sequence(data, value=0):
  max_length = max([len(d) for d in data])
  padded = []
  for datum in data:
    padded.append(
        jnp.pad(
            datum, (0, max_length - len(datum)),
            mode='constant',
            constant_values=value))
  return jnp.stack(padded)


rng_seq = hk.PRNGSequence(42)
np.random.seed(42)

net = hk.transform(lambda x, y, training: build_model()
                   (x, y, training=training))


def load_batch():
  data = []
  for _ in range(args.batch_size):
    data.append(
        utils.generate_toy_data(
            num_symbols=args.num_symbols, num_segments=args.num_segments))
  lengths = jnp.array(list(map(len, data)), dtype=jnp.int32)
  lengths = lengths
  return pad_sequence(data), lengths

inputs, lengths = load_batch()
params = net.init(next(rng_seq), inputs, lengths, training=True)
opt = optax.adam(learning_rate=args.learning_rate)
opt_state = opt.init(params)


def loss(params, rng, inputs, lengths, training):
  outputs = net.apply(params, rng, inputs, lengths, training)
  loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)
  return loss, (loss, nll, kl_z, kl_b)


def eval(params, rng, inputs, lengths):
  outputs = net.apply(params, rng, inputs, lengths, False)
  acc, rec = utils.get_reconstruction_accuracy(inputs, outputs, args)
  return acc, rec


def update(params, rng, opt_state, inputs, lengths):
  grads, (l, nll, kl_z, kl_b) = jax.grad(
      loss, has_aux=True)(params, rng, inputs, lengths, True)
  updates, opt_state = opt.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state, (l, nll, kl_z, kl_b)


loss_fn = jax.jit(loss, static_argnums=(4,))
eval_fn = eval
update_fn = jax.jit(update)

# Train model.
print('Training model...')
for step in range(args.iterations):
  data = None
  rec = None
  batch_loss = 0
  batch_acc = 0

  # Generate data.
  inputs, lengths = load_batch()

  params, opt_state, (l, nll, kl_z, kl_b) = update_fn(params, next(rng_seq),
                                                      opt_state, inputs,
                                                      lengths)

  if step % args.log_interval == 0:
    # Run evaluation.
    acc, rec = eval_fn(params, next(rng_seq), inputs, lengths)

    # Accumulate metrics.
    batch_acc += acc.item()
    batch_loss += nll.item()
    print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(
        step, batch_loss, batch_acc))
    print('input sample: {}'.format(inputs[-1, :lengths[-1] - 1]))
    print('reconstruction: {}'.format(rec[-1]))
