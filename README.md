# SVD-AE

This code implements [SVD-AE](https://arxiv.org/abs/2405.04746), which is based on the implementation of [∞-AE](https://github.com/noveens/infinite_ae_cf).

Before running the code, please download the data from [here](https://drive.google.com/file/d/1cuhQw1aR9BEwutK3svKtL_-CGmcIPOiX/view?usp=sharing) and unzip it.

To run SVD-AE, use the following commands:

## ML-1M
python main.py --dataset ml-1m --k 148

## ML-10M
python main.py --dataset ml-10m --k 427

## Gowalla
python main.py --dataset gowalla --k 1194

## Yelp
python main.py --dataset yelp2018 --k 1267
