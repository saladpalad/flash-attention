# Implementation of forward pass of Flash Attention V1

## Setup
```
pip install -r requirements.txt
sudo apt update
sudo apt-get install ninja-build
```

## Run
```
python3 main.py
```

## Results on 3060 Ti
```
Starting attention computations...
Naive attention time: 50.9389 ms
Flash attention time: 0.6687 ms
Speedup: 76.18x
Attention values sanity check: True
```
