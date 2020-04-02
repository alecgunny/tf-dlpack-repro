import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_virtual_device_configuration(
  gpu,
  [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
)

import numba
import cudf
import cupy as cp
import string


def get_mem_info():
  return numba.cuda.current_context().get_memory_info()


def get_external_mem_delta(func):
  '''
  quick wrapper function for measuring the change in free
  device memory outside of a function call. Returns this
  along with the output of the function
  '''
  def wrapper(*args, **kwargs):
    init_free_mem = get_mem_info().free
    output = func(*args, **kwargs)
    external_mem_delta = init_free_mem - get_mem_info().free
    return external_mem_delta, output
  return wrapper


@get_external_mem_delta
def make_df(num_rows=2**24, num_columns=5, export_to=None):
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


@get_external_mem_delta
def loop_make_df(loops=10, num_rows=2**24, num_cols=5, export_to=None):
  '''
  iteratively measure the external and internal memory usage of
  creating and possibly exporting a cudf dataframe
  '''
  external_deltas, internal_deltas = [], []
  for _ in range(loops):
    external_delta, internal_delta = make_df(num_rows, num_cols, export_to)
    external_deltas.append(external_delta)
    internal_deltas.append(internal_delta)
  return external_deltas, internal_deltas


@get_external_mem_delta
def initialize_tensorflow():
  a = tf.constant(0)

# initialize tf eager context
print('Free memory before TensorFlow initialization: {} B'.format(
  get_mem_info().free))
print('Mem delta from TensorFlow initialization: {} B\n'.format(
  initialize_tensorflow()[0]))

# let cudf initialize
print('Free memory before CuDf initialization: {} B'.format(
  get_mem_info().free))
print('Mem delta from CuDf initialization: {} B\n'.format(
  make_df()[0]))

# run loops
print('Free memory before loops: {} B'.format(get_mem_info().free))
cudf_loop_delta, (cudf_external_deltas, cudf_internal_deltas) = loop_make_df()
print('Total mem delta from looping cudf creation 10 times: {} B\n'.format(
  cudf_loop_delta))

dlpack_loop_delta, (dlpack_external_deltas, dlpack_internal_deltas) = \
  loop_make_df(export_to='dlpack')
print('Total mem delta from looping dlpack creation 10 times: {} B\n'.format(
  dlpack_loop_delta))

tf_loop_delta, (tf_external_deltas, tf_internal_deltas) = \
  loop_make_df(export_to='tf')
print('Total mem delta from looping tf creation 10 times: {} B\n'.format(
  tf_loop_delta))

# make sure that we can account for all the memory lost in tf loops
assert all([ext_d == int_d for ext_d, int_d in zip(tf_external_deltas, tf_internal_deltas)])

# verify that it's equal to the memory we expect the float64 tensors
# we created to use
assert all([(d / 8) == 5*(2**24) for d in tf_external_deltas])
