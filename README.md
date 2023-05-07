# Feature Interaction in Recommendation Systems

This project created a benchmark comparing

- 6 feature interaction methods: Factorization Machines (FM), Product-based Neural Network (PNN), DeepFM, Deep Cross Network (DCN), Deep Cross Network v2 (DCNv2), Attention Factorization Machine (AFM)

- on 2 datasets: Criteo, Avazu.


The github is based on: https://github.com/fuyuanlyu/OptInter

***Contribution:*** Three models were added: DCN, DCNv2, Attention FM (AFM)

***Note:***
To run the demo or training, the fastest way is to download the full folder including original datasets and dataset after preprocess from google drive:
https://drive.google.com/drive/folders/1a40pNMy1W2TD0QZwdU-dVjGYZRzdf2vr?usp=share_link


## Running Demo
```
cd demo
streamlit run app.py
```

## Running Model
### Enviroment
```
pip install -r model/requirements.txt
```

### Dataset
Criteo dataset: https://www.kaggle.com/datasets/mrkmakr/criteo-dataset

Avazu dataset: https://www.kaggle.com/c/avazu-ctr-prediction/data

For Criteo dataset, copy the train.txt file under datasets/Criteo and rename it to full.txt.

For Avazu dataset, copy the train.csv file under datasets/Avazu and rename it to full.csv.

```
mkdir datasets
mkdir datasets/Criteo
mkdir datasets/Avazu
```

### Data processing
To speed up, I only use 8% data as training set and 2% data as test set.

```
cd model
nohup python preprocess/criteo.py &> res/pre_criteo &
nohup python preprocess/avazu.py &> res/pre_avazu &
```

### Model training
The learning rate all set to 0.001 except AFM (0.01) which is slow to converge.

```
cd model
nohup python learn/CriteoTrain.py --model FM --gpu 0 &> res/train_criteo &
nohup python learn/AvazuTrain.py --model FM --gpu 0 &> res/train_avazu &

nohup python learn/CriteoTrain.py --model IPNN --gpu 0 &> res/train_criteo_ipnn &
nohup python learn/AvazuTrain.py --model IPNN --gpu 0 &> res/train_avazu_ipnn &

nohup python learn/CriteoTrain.py --model DeepFM --gpu 0 &> res/train_criteo_deepfm &
nohup python learn/AvazuTrain.py --model DeepFM --gpu 0 &> res/train_avazu_deepfm &

nohup python learn/CriteoTrain.py --model DCN --gpu 0 &> res/train_criteo_dcn &
nohup python learn/AvazuTrain.py --model DCN --gpu 0 &> res/train_avazu_dcn &

nohup python learn/CriteoTrain.py --model DCNv2 --gpu 0 &> res/train_criteo_dcnv2 &
nohup python learn/AvazuTrain.py --model DCNv2 --gpu 0 &> res/train_avazu_dcnv2 &

nohup python learn/CriteoTrain.py --model AFM --gpu 0 --lr 0.01 &> res/train_criteo_afm &
nohup python learn/AvazuTrain.py --model AFM --gpu 0 --lr 0.01 &> res/train_avazu_afm &

```


### Comparing Embedding Size

```
cd model
nohup python learn/CriteoTrain.py --model IPNN --gpu 0 --orig_embedding_dim 5 &> res/train_criteo_ipnn_d5 &
nohup python learn/CriteoTrain.py --model IPNN --gpu 0 --orig_embedding_dim 10 &> res/train_criteo_ipnn_d10 &
nohup python learn/CriteoTrain.py --model IPNN --gpu 0 --orig_embedding_dim 20 &> res/train_criteo_ipnn_d20 &
nohup python learn/CriteoTrain.py --model IPNN --gpu 0 --orig_embedding_dim 80 &> res/train_criteo_ipnn_d80 &


nohup python learn/AvazuTrain.py --model IPNN --gpu 0 --orig_embedding_dim 5 &> res/train_avazu_ipnn_d5 &
nohup python learn/AvazuTrain.py --model IPNN --gpu 0 --orig_embedding_dim 10 &> res/train_avazu_ipnn_d10 &
nohup python learn/AvazuTrain.py --model IPNN --gpu 0 --orig_embedding_dim 20 &> res/train_avazu_ipnn_d20 &
nohup python learn/AvazuTrain.py --model IPNN --gpu 0 --orig_embedding_dim 80 &> res/train_avazu_ipnn_d80 &

```

#### Appendix

Attention FM refered to: 

https://github.com/hexiangnan/attentional_factorization_machine

https://github.com/rixwew/pytorch-fm
