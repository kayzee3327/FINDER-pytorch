# FINDER-pytorch Environment
```shell
conda create -n FINDER-pytorch 3.11
conda activate FINDER-pytorch
conda install tqdm cython networkx scipy numpy pandas
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
# Build
```shell 
python setup.py build_ext -i
```

# Note
   