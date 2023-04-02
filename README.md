# start point:
Paper: Memorize, Factorize, or be Naive: Learning Optimal Feature Interaction Methods for CTR Prediction 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9835208&tag=1

### Data processing
```
cd model/OptInter-master
nohup python preprocess/criteo.py &> res/pre_criteo &
nohup python preprocess/avazu.py &> res/pre_avazu &
```

