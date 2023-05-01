## Feature Interaction in Recommendation Systems

This project created a benchmark comparing

- 6 feature interaction methods: Factorization Machines (FM), Product-based Neural Network (PNN), DeepFM, Deep Cross Network (DCN), Deep Cross Network v2 (DCNv2), Attention Factorization Machine (AFM)

- on 2 datasets: Criteo, Avazu.


The github is based on: https://github.com/fuyuanlyu/OptInter

***Contribution:*** Three models were added: DCN, DCNv2, Attention FM (AFM)

### Enviroment
```
pip install -r model/requirements.txt
```


### Folder
```
cd /content/drive/MyDrive/Lehigh/Courses/'DSCI 441 Statistical and Machine Learning'/project/Recommendation_Systems/model/

cd /Users/cancan/caz322@lehigh.edu\ -\ Google\ Drive/My\ Drive/Lehigh/Courses/DSCI\ 441\ Statistical\ and\ Machine\ Learning/project/Recommendation_Systems
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

#### Appendix

Attention FM refered to: 

https://github.com/hexiangnan/attentional_factorization_machine

https://github.com/rixwew/pytorch-fm
