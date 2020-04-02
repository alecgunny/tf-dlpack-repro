## Example Usage
```
docker build -t rapids-tf-nightly .
docker run --rm -it --gpus 1 rapids-tf-nightly
```
## Example Output
Cleaned up a bit to give you the gist
``
Free memory before TensorFlow initialization: 15998058496 B
Mem delta from TensorFlow initialization: 465567744 B

Free memory before CuDf initialization: 15532490752 B
Mem delta from CuDf initialization: 675282944 B

Free memory before loops: 14857207808 B
Total mem delta from looping cudf creation 10 times: 0 B

Total mem delta from looping dlpack creation 10 times: 0 B

Total mem delta from looping tf creation 10 times: 6710886400 B
```
