import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
from utils import generate_cate_dict, process_cont, process_cate, generate_comb_dict, process_comb
from utils import read_data, write_to_tfrecord

def train_test_split_avazu(source_dir, target_dir, ratio=0.8):
    all_file = os.path.join(source_dir, 'full.csv')
    train_file = os.path.join(target_dir, 'train.txt')
    test_file = os.path.join(target_dir, 'test.txt')
    if os.path.exists(train_file):
        return

    df = pd.read_csv(all_file)
    day = df['hour'] // 100
    weekday = day % 7
    hour = df['hour'] % 100
    del df['hour']
    del df['id']
    df['day'] = day
    df['hour'] = hour
    df['weekday'] = weekday

    #index = int(ratio*(len(df)))
    index_train = int(ratio*len(df)*0.1)
    index_test = int((1-ratio)*len(df)*0.1)
    
    df = shuffle(df)
    train_df = df.iloc[:index_train, :]
    test_df = df.iloc[-index_test:, :]
    print ('train_df', train_df.shape)
    print ('test_df', test_df.shape)
    
    train_df.to_csv(train_file, sep='\t', header=None, index=False)
    test_df.to_csv(test_file, sep='\t', header=None, index=False)

def generate_orig_24(source_dir, target_dir, X=5):
    CONTS = 0
    CATES = 24
    COMBS = 0
    cont_cols = ["cont_{}".format(i+1) for i in range(CONTS)]
    cate_cols = ["cate_{}".format(i+1) for i in range(CATES)]

    # Read data
    train_X, train_y = read_data(source_dir, CONTS, CATES, name='train')
    test_X,  test_y  = read_data(source_dir, CONTS, CATES, name='test')

    # Generate dictionary for category feature 
    dict_dir = os.path.join(target_dir, 'X_'+str(X), 'dict_cate')
    if not os.path.exists(os.path.join(target_dir, 'X_'+str(X))):
        os.mkdir(os.path.join(target_dir, 'X_'+str(X)))
    if not os.path.exists(dict_dir):
        os.mkdir(dict_dir)
        
    generate_cate_dict(target_dir, train_X, cate_cols, cont_cols, X)

    # Applying dictionary to categorical feature
    cate_train_X = process_cate(train_X, cont_cols, cate_cols, dict_dir)
    cate_test_X = process_cate(test_X, cont_cols, cate_cols, dict_dir)

    # Maximum id
    print("maximun id: {}".format(np.max(cate_train_X)))

    # Write train data to tfrecord
    orig_dir = os.path.join(target_dir, 'X_' + str(X), 'orig_24')
    write_to_tfrecord(cate_train_X, train_y, orig_dir, CONTS, CATES, COMBS, name='train')
    write_to_tfrecord(cate_test_X, test_y, orig_dir, CONTS, CATES, COMBS, name='test')

def generate_comb_276(source_dir, target_dir, X=5, Y=1):
    CONTS = 0
    CATES = 24
    COMBS = 276
    cont_cols = ["cont_{}".format(i+1) for i in range(CONTS)]
    cate_cols = ["cate_{}".format(i+1) for i in range(CATES)]

    # Read data
    train_X, train_y = read_data(source_dir, CONTS, CATES, name='train')
    test_X,  test_y  = read_data(source_dir, CONTS, CATES, name='test')
    print ('train_X', train_X.shape)
    print ('test_X', test_X.shape)

    # Generate selected pairs
    fields = np.arange(CATES)
    selected_pairs = []
    for i in range(len(fields)):
        for j in range(i+1, len(fields)):
            selected_pairs.append((fields[i], fields[j]))
    print ('selected pairs generated...')

    # Generate combinational dictionary
    dict_dir_comb = os.path.join(target_dir, 'X_' + str(X), 'dict_comb_276')
    generate_comb_dict(dict_dir_comb, train_X, selected_pairs, Y=Y)

    # Applying dictionary to categorical feature
    dict_dir = os.path.join(target_dir, 'X_' + str(X), 'dict_cate')
    cate_train_X = process_cate(train_X, cont_cols, cate_cols, dict_dir)
    cate_test_X = process_cate(test_X, cont_cols, cate_cols, dict_dir)

    # Maximum Orig id
    print("maximun orig id: {}".format(np.max(cate_train_X)))

    # Write train data to tfrecord
    comb_train_X = process_comb(train_X, dict_dir_comb, selected_pairs, Y)
    comb_test_X = process_comb(test_X, dict_dir_comb, selected_pairs, Y)

    # Maximum Comb id
    print("maximun comb id: {}".format(np.max(comb_train_X)))

    # concatenate
    final_train_X = np.concatenate([cate_train_X, comb_train_X], axis=1)
    final_test_X = np.concatenate([cate_test_X, comb_test_X], axis=1)
    print ('final_train_X', final_train_X.shape)
    print ('final_test_X', final_test_X.shape)
    
    # Write train data to tfrecord
    print ('writing...')
    orig_dir = os.path.join(target_dir, 'X_' + str(X), 'comb_276_Y_' +str(Y))
    write_to_tfrecord(final_train_X, train_y, orig_dir, CONTS, CATES, COMBS, name='train', partnum=500000)
    write_to_tfrecord(final_test_X, test_y, orig_dir, CONTS, CATES, COMBS, name='test', partnum=500000)

def main():
    #prefix = '/home/jupyter-caz322/lehigh_courses/DSCI_441/Recommendation_Systems/'
    prefix = '/content/drive/MyDrive/Lehigh/Courses/DSCI 441 Statistical and Machine Learning/project/Recommendation_Systems/'
    source_dir = prefix + 'datasets/Avazu'
    target_dir = prefix + 'datasets/Avazu-new'
    os.makedirs(target_dir, exist_ok=True)
    #train_test_split_avazu(source_dir, source_dir, ratio=0.8)
    generate_orig_24(source_dir, target_dir)
    #generate_comb_276(source_dir, target_dir)

if __name__ == "__main__":
    main()
