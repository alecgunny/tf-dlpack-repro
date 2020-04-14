import tensorflow as tf
tf.config.set_logical_device_configuration(
  tf.config.list_physical_devices('GPU')[0],
  [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
)

import torch
from torch.utils.dlpack import to_dlpack as pt_to_dlpack
from torch.utils.dlpack import from_dlpack as pt_from_dlpack

import numba
import cudf
import cupy as cp
import string
import argparse
import warnings


def get_mem_info():
  return numba.cuda.current_context().get_memory_info()


def get_name(func, **kwargs):
  if kwargs.get('name') is not None:
    return kwargs['name']

  else:
    if kwargs.get('loops') is None:
      if 'export_to' not in kwargs:
        return func.__name__
      else:
        return '{} as {}'.format(
          func.__name__, kwargs['export_to'] or 'cudf')
    else:
      return '{} loops exporting from {} to {}'.format(
        kwargs['loops'], 
        kwargs.get('make_in') or 'cudf',
        kwargs['export_to'] or kwargs.get('make_in') or 'cudf')


def report_external_mem_delta(func):
  '''
  quick wrapper function for measuring the change in free
  device memory outside of a function call. Returns this
  along with the output of the function
  '''
  def wrapper(*args, **kwargs):
    no_report = kwargs.get('no_report', False)
    if not no_report:
      init_free_mem = get_mem_info().free
      print('Free device memory before {}: {} B'.format(
        get_name(func, **kwargs), init_free_mem)
      )

    output = func(*args, **kwargs)

    if not no_report:
      external_mem_delta = init_free_mem - get_mem_info().free
      print('Free memory delta from {}: {} B\n'.format(
        get_name(func, **kwargs), external_mem_delta)
      )
      if kwargs.get('make_in') == 'pt':
        print('PyTorch current reserved bytes: {} B\n'.format(
          torch.cuda.memory_stats()['reserved_bytes.large_pool.current'])
        )

    return output
  return wrapper


def cudf_to_dlpack(column):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return column.to_dlpack()


@report_external_mem_delta
def make_data(
    make_in='cudf',
    num_rows=2**24,
    num_columns=10,
    export_to=None,
    **kwargs):
  '''
  Measures the free device memory used by a random
  dataframe with the given numbers of rows and columns,
  including an optional step of converting to a dlpack
  capsule, which can then be optionall converted to a
  tensorflow tensor
  '''
  init_free_mem = get_mem_info().free

  # create a dummy dataframe
  columns = string.ascii_letters[:num_columns]
  if make_in == 'cudf':
    df = cudf.DataFrame(
      {column: cp.random.randn(num_rows).astype(cp.float32) for column in columns}
    )
    df = {column: df[column] for column in columns}
  elif make_in == 'pt':
    df = {column: torch.randn(num_rows, device='cuda:0') for column in columns}
  else:
    raise ValueError('make_in must be cudf or pt')

  # optionally export to dlpack and then from there to tf or pt
  if export_to is not None:
    export_fn = cudf_to_dlpack if make_in == 'cudf' else pt_to_dlpack

    try:
      import_fn = {
        'tf': tf.experimental.dlpack.from_dlpack,
        'pt': pt_from_dlpack,
        'dlpack': lambda x: x
      }[export_to]
    except KeyError:
      raise ValueError('Unrecognized export lib {}'.format(export_to))

    tensors = {column: import_fn(export_fn(x)) for column, x in df.items()}

  # return the delta in free device memory
  return init_free_mem - get_mem_info().free


@report_external_mem_delta
def loop_make_data(
    loops=10,
    num_rows=2**24,
    num_columns=5,
    make_in='cudf',
    export_to=None,
    **kwargs):
  '''
  iteratively measure the external and internal memory usage of
  creating and possibly exporting a cudf dataframe
  '''
  internal_deltas = []
  for _ in range(loops):
    internal_deltas.append(
      make_data(make_in, num_rows, num_columns, export_to, no_report=True)
    )
  return internal_deltas


@report_external_mem_delta
def initialize_tensorflow(**kwargs):
  a = tf.random.normal((1,))


@report_external_mem_delta
def initialize_pytorch(**kwargs):
  a = torch.randn(1, device='cuda:0')


def main(flags):
  initialize_tensorflow(name='TensorFlow initialization')
  initialize_pytorch(name='PyTorch initialization')

  make_data(name='cuDF initialization', export_to=None, **flags)

  cudf_internal_deltas = loop_make_data(
    make_in='cudf', export_to=None, **flags)

  cudf_to_dlpack_internal_deltas = loop_make_data(
    make_in='cudf', export_to='dlpack', **flags)

  cudf_to_pt_internal_deltas = loop_make_data(
    make_in='cudf', export_to='pt', **flags)

  pt_internal_deltas = loop_make_data(
    make_in='pt', export_to=None, **flags)

  pt_to_dlpack_internal_deltas = loop_make_data(
    make_in='pt', export_to='dlpack', **flags)

  pt_to_tf_internal_deltas = loop_make_data(
    make_in='pt', export_to='tf', **flags)

  cudf_to_tf_internal_deltas = loop_make_data(
    make_in='cudf', export_to='tf', **flags)

  # check that the memory lost matches with what we would expect
  # our tensors to occupy
  total_tf_mem_lost = sum(cudf_to_tf_internal_deltas)
  total_expected_mem_lost = (
    flags['loops']*flags['num_rows']*flags['num_columns']
    *4 # because float32
    *2 # because we have TF object and cuDF
  )
  assert total_tf_mem_lost == total_expected_mem_lost


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--loops',
    type=int,
    default=10,
    help='Number of iterations of data creation to perform'
  )
  parser.add_argument(
    '--num_rows',
    type=int,
    default=2**24,
    help='Number of rows of data to create at each iteration'
  )
  parser.add_argument(
    '--num_columns',
    type=int,
    default=10,
    help='Number of columns of data to create at each iteration'
  )
  FLAGS = parser.parse_args()
  main(vars(FLAGS))

