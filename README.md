# HSI Toolkit

## Prerequisites

### Environment

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install mmengine icecream einops clearml transformers click scipy h5py timm fvcore
```

### Datasets

#### ICVL

**1. Download**

Original: [BGU ICVL Hyperspectral Dataset](https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html#)

Test set with static noise: [Baidu Pan](https://pan.baidu.com/s/1BkNYhb9CBtXnKsQjNwYFyg#list/path=%2F) (Pass: HSIR) -> Denoise -> ICVL_Test

Training set: [Baidu Pan](https://pan.baidu.com/s/1BkNYhb9CBtXnKsQjNwYFyg#list/path=%2F) (Pass: HSIR) -> Dataset -> IVCL

**2. Preprocessing**

Slicing training set (place the downloaded `Baidu Pan` training set folder in the `data` folder):

```
python slice_icvl_train.py --source_folder data/IVCL/IVCLmat --out_folder data/icvl/imgs --res 128 --color_bit_depth 12
```

Slicing test Set (place the downloaded `Baidu Pan` test set folder in the `data` folder):
```
python slice_icvl_test.py --source_folder data/ICVL_Test/icvl_512_50 --out_folder data/icvl/test_imgs --res 64
```

**3. File System Architecture**

```
root
└── data
    ├── icvl
    │   ├── imgs
    │   │   ├── BGU_0403-1419-1_0_0.npy
    │   │   └── ...
    │   └── splits
    │   │   ├── icvl_train_half_part1.txt
    │   │   └── icvl_train_half_part2.txt
    │   └── test_imgs_64
    │       ├── bulb_0822-0909_0_0.npy
    │       └── ...
    ├── icvl_test
    │   ├── icvl_512_30
    │   ├── icvl_512_50
    │   ├── icvl_512_70
    │   └── ...
    ├── Others
    └── ...
```

If you need a custom folder structure, modify the `folder_arch` dictionary in the `gsettings.py` file to fit your folder architecture.
Additionally, ensure these folders have read and write permissions.

## Quick Start

### Training

```bash
python train.py --model_arch_name hsdt_s --train_dataset_name icvl_train_half_part1 --train_noise_type gaussian50 --val_dataset_name icvl_test --val_noise_type gaussian50 --optim_sche_strategy hsdtstage1 --batch_size=4 --max_epoches 30 --save_cp_interval 10 --lr 1e-3
```

### Testing

```bash
python val.py --model_arch_name hsdt_s --dataset_name icvl_test --noise_types gaussian30,gaussian50,gaussian70 --num_bands 31 --ckpt_path checkpoints/hsdt_s_gaussian.pth
```
