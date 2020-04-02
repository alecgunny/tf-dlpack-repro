## Example Usage
```
docker build -t rapids-tf-nightly .
docker run --rm -it --gpus 1 rapids-tf-nightly python expt.py
```
## Example Output
Cleaned up a bit to give you the gist
```
Free device memory before TensorFlow initialization: 15943532544 B
Free memory delta from TensorFlow initialization: 467664896 B

Free device memory before cuDF initialization: 15475867648 B
Free memory delta from cuDF initialization: 1346371584 B

Free device memory before 10 loops exporting to cudf: 14129496064 B
Free memory delta from 10 loops exporting to cudf: 0 B

Free device memory before 10 loops exporting to dlpack: 14129496064 B
Free memory delta from 10 loops exporting to dlpack: 0 B

Free device memory before 10 loops exporting to tf: 14129496064 B
Free memory delta from 10 loops exporting to tf: 13421772800 B
```
