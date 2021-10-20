# POINT CLOUD GEOMETRY COMPRESSION VIA NEURAL GRAPH SAMPLING
2021.5.20 Our paper has been accepted by **ICIP2021**!

## Requirments

- open3d 0.9.0.0

- python 3.7

- cuda 10.2 

- pytorch 1.7.0

## Usage

### Training
```shell
 python train.py --dataset_path='training_dataset_path'
```

### Testing
```shell
sudo chmod 777 utils/pc_error
python test.py --dataset_path='training_dataset_path'
```
