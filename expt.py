import tensorflow as tf
tf.config.set_logical_device_configuration(
  tf.config.list_physical_devices('GPU')[0],
  [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
)

import numba
import cudf
import cupy as cp
import string
import argparse


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
      return '{} loops exporting to {}'.format(
        kwargs['loops'], kwargs['export_to'] or 'cudf')


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

    return output
  return wrapper


@report_external_mem_delta
def make_df(num_rows=2**24, num_columns=10, export_to=None, **kwargs):
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
  df = cudf.DataFrame(
    {column: cp.random.randn(num_rows) for column in columns}
  )

  # optionally export to dlpack and then from there to tf
  if export_to is not None:
    assert export_to in ('tf', 'dlpack')
    capsules = {column: df[column].to_dlpack() for column in columns}
    if export_to == 'tf':
      tensors = {
        column: tf.experimental.dlpack.from_dlpack(capsule)
        for column, capsule in capsules.items()
      }

  # return the delta in free device memory
  return init_free_mem - get_mem_info().free


@report_external_mem_delta
def loop_make_df(
    loops=10, num_rows=2**24, num_columns=5, export_to=None, **kwargs):
  '''
  iteratively measure the external and internal memory usage of
  creating and possibly exporting a cudf dataframe
  '''
  internal_deltas = []
  for _ in range(loops):
    internal_deltas.append(
      make_df(num_rows, num_columns, export_to, no_report=True)
    )
  return internal_deltas


@report_external_mem_delta
def initialize_tensorflow(**kwargs):
  a = tf.constant(0)


def main(flags):
  initialize_tensorflow(name='TensorFlow initialization')
  make_df(name='cuDF initialization', export_to=None, **flags)

  cudf_internal_deltas = loop_make_df(
    export_to=None, **flags)

  dlpack_internal_deltas = loop_make_df(
    export_to='dlpack', **flags)

  tf_internal_deltas = loop_make_df(
    export_to='tf', **flags)

  # check that the memory lost matches with what we would expect
  # our tensors to occupy
  total_tf_mem_lost = sum(tf_internal_deltas)
  total_expected_mem_lost = (
    flags['loops']*flags['num_rows']*flags['num_columns']*8 # float64
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

