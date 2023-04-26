The github is based on:
Paper: Memorize, Factorize, or be Naive: Learning Optimal Feature Interaction Methods for CTR Prediction 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9835208&tag=1
```
cd /content/drive/MyDrive/Lehigh/Courses/'DSCI 441 Statistical and Machine Learning'/project/Recommendation_Systems/model/OptInter-master
```
### Data processing
To speed up, I only use 8% data as training set and 2% data as test set.

```
cd model/OptInter-master
nohup python preprocess/criteo.py &> res/pre_criteo &
nohup python preprocess/avazu.py &> res/pre_avazu &
```

### Model training

Attention FM refered to: 
https://github.com/hexiangnan/attentional_factorization_machine
https://github.com/rixwew/pytorch-fm


```
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
