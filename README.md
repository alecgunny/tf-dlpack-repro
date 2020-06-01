Quick repro showing memory leakage issues when using TensorFlow experimental `from_dlpack` function with cuDF dataframes. Use case of interest is a function which internally creates a cudf dataframe, then exports its columns to TensorFlow tensors (which would normally be returned). Each successive call to `from_dlpack` seems to increase memory usage, which doesn't happen when just creating the dataframe or even exporting it to dlpack.

Here we measure the changes in free memory after successive calls to each framework *after* the function creating and exporting the data has exited. When we just create a dataframe or even export it to dlpack, the memory is released once the function exits and we see no change in GPU free memory. However, each `from_dlpack` call permanently reduces available memory  by an amount corresponding precisely to the size of the data created (see the `assert` statement at the bottom of `expt.main`).

## Example Usage
```
docker build -t rapids-tf2.2 .
docker run --rm -it --gpus 1 rapids-tf2.2 python expt.py
```
## Example Output
Cleaned up a bit to give you the gist
```
Free device memory before TensorFlow initialization: 33726332928 B
Free memory delta from TensorFlow initialization: 8734638080 B

Free device memory before PyTorch initialization: 24991694848 B
Free memory delta from PyTorch initialization: 1134559232 B

Free device memory before cuDF initialization: 23857135616 B
Free memory delta from cuDF initialization: 2097152 B

Free device memory before 10 loops exporting from cudf to cudf: 23855038464 B
Free memory delta from 10 loops exporting from cudf to cudf: 0 B

Free device memory before 10 loops exporting from cudf to dlpack: 23855038464 B
Free memory delta from 10 loops exporting from cudf to dlpack: 0 B

Free device memory before 10 loops exporting from cudf to pt: 23855038464 B
Free memory delta from 10 loops exporting from cudf to pt: 0 B

Free device memory before 10 loops exporting from pt to pt: 23855038464 B
Free memory delta from 10 loops exporting from pt to pt: 671088640 B

PyTorch current reserved bytes: 671088640 B

Free device memory before 10 loops exporting from pt to dlpack: 23183949824 B
Free memory delta from 10 loops exporting from pt to dlpack: 0 B

PyTorch current reserved bytes: 671088640 B

Free device memory before 10 loops exporting from pt to tf: 23183949824 B
Free memory delta from 10 loops exporting from pt to tf: 6039797760 B

PyTorch current reserved bytes: 6710886400 B

Free device memory before 10 loops exporting from cudf to tf: 17144152064 B
Free memory delta from 10 loops exporting from cudf to tf: 6710886400 B
```
Let's go through this section by section:
```
Free device memory before 10 loops exporting from cudf to cudf: 23855038464 B
Free memory delta from 10 loops exporting from cudf to cudf: 0 B

Free device memory before 10 loops exporting from cudf to dlpack: 23855038464 B
Free memory delta from 10 loops exporting from cudf to dlpack: 0 B

Free device memory before 10 loops exporting from cudf to pt: 23855038464 B
Free memory delta from 10 loops exporting from cudf to pt: 0 B
```
cuDF releases memory as soon as it's not needed, which makes this all pretty reasonable. Even when we export to PyTorch using `from_dlpack`, this just points to the original cuDF memory rather than managing the memory in PyTorch itself, so we see no additional allocation.
```
Free device memory before 10 loops exporting from pt to pt: 23855038464 B
Free memory delta from 10 loops exporting from pt to pt: 671088640 B

PyTorch current reserved bytes: 671088640 B
```
Next, when we iteratively build PyTorch tensors inside our data creation function, we lose exactly the amount of memory corresponding to the data in *one* call to the function (2\*\*24 rows x 10 columns x 4 bytes per float32). Why? Because PyTorch's cached allocator reserves this data on the first iteration. Once that iteration is done, it frees the memory for other PyTorch usage, but keeps it reserved on the GPU. Then successive calls to our data creation function iteratively allocate and then free this same block of reserved memory.
```
Free device memory before 10 loops exporting from pt to dlpack: 23183949824 B
Free memory delta from 10 loops exporting from pt to dlpack: 0 B

PyTorch current reserved bytes: 671088640 B
```
This is why we see no delta here: this loop continues to leverage the same chunk of reserved memory, even when exporting to a dlpack capsule, and so no new memory is used.
```
Free device memory before 10 loops exporting from pt to tf: 23183949824 B
Free memory delta from 10 loops exporting from pt to tf: 6039797760 B

PyTorch current reserved bytes: 6710886400 B
```
This is where things get interesting. When we export a PyTorch tensor to TensorFlow via dlpack, the total memory delta over 10 loops is equal to the data used in total by 9 loops. Why 9? Well, the first iteration uses that same reserved chunk of memory we were using before. However, evidently something about the export to TensorFlow keeps PyTorch from freeing this memory, and so successive data creation calls need to reserve more data to accommodate the new tensors. Note that PyTorch cops to this by telling us that its current reserved byte usage is now equal to 10x the amount that it was before.
```
Free device memory before 10 loops exporting from cudf to tf: 17144152064 B
Free memory delta from 10 loops exporting from cudf to tf: 6710886400 B
```
The behavior exhibited by exporting from cuDF to TensorFlow via dlpack, then, is consistent with what we would expect given the PyTorch behavior detailed above. Something about TensorFlow's `from_dlpack` keeps the memory from getting released, and so we lose free memory in each data creation call.
