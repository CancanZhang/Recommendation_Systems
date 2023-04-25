The github is based on:
Paper: Memorize, Factorize, or be Naive: Learning Optimal Feature Interaction Methods for CTR Prediction 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9835208&tag=1
```
cd /content/drive/MyDrive/Lehigh/Courses/DSCI 441 Statistical and Machine Learning/project/Recommendation_Systems/model/OptInter-master
```
### Data processing
To speed up, I only use 8% data as training set and 2% data as test set.

```
cd model/OptInter-master
nohup python preprocess/criteo.py &> res/pre_criteo &
nohup python preprocess/avazu.py &> res/pre_avazu &
```

### Model training
```
nohup python learn/CriteoTrain.py --model FM --gpu 0 &> res/train_criteo &
nohup python learn/AvazuTrain.py --model FM --gpu 0 &> res/train_avazu &



nohup python learn/CriteoTrain.py --model DCN --gpu 0 &> res/train_criteo_dcn &
nohup python learn/AvazuTrain.py --model FM --gpu 0 &> res/train_avazu &
```
