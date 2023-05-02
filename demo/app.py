import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('../model')

# markdown
st.markdown('DSCI 441 Final Project')

# 设置网页标题
st.title('Feature Interaction in Recommendation Systems Demo')

@st.cache_data
def load_data(dataset,nrows=None):
    if dataset == 'Avazu':
        h =['id','label (click)','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type']
        for i in range(14,22):
            h.append('C'+str(i))
        data = pd.read_csv('../datasets/Avazu/train.txt', sep='\t', header=None, names=h)
    elif dataset == 'Criteo':
        h = ['label (click)']
        for i in range(1,14):
            h.append('I'+str(i))
        for i in range(1,27):
            h.append('C'+str(i))
        data = pd.read_csv('../datasets/Criteo/train.txt', sep='\t', header=None, names=h)
    else:
        raise Exception('Wrong dataset name')
    if nrows:
        data = data[:nrows]
    return data

st.header('1. Datasets')
tab1, tab2 = st.tabs(['Avazu','Criteo'])
data_avazu = load_data('Avazu',6)
data_criteo = load_data('Criteo',6)

with tab1:
    st.write('C1: anonymized categorical variable.')
    st.write('C14-C21: anonymized categorical variable.')

    st.dataframe(data_avazu)
with tab2:
    st.write('Label: Target variable that indicates if an ad was clicked (1) or not (0).')
    st.write('I1-I13: A total of 13 columns of integer features, mostly count features.')
    st.write('C1-C26: A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.')
    st.dataframe(data_criteo)

st.header('2. Models')
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['FM','PNN','DeepFM','DCN','DCNv2','AFM'])
with tab1:
    st.write("Factorization Machines (FM) embeds features into dense features and use inner product to capture feature interactions:")
    st.markdown("$$y_{FM} = w_0+\sum_{i=1}^{d}w_ix_i + \sum_{i=1}^{d} \sum_{j=i+1}^{d} v_i^T v_j x_i x_j.$$")

with tab2:
    st.write("Product-based Neural Network (PNN) concate features interaction results with features instead of adding these two parts.")
    image = Image.open('figs/PNN.png')
    st.image(image, caption='PNN architecture')

with tab3:
    st.write("DeepFM uses FM layer to capture low-order feature interactions while dnn layer to capture higher-order feature interactions")
    image = Image.open('figs/deepFM.png')
    st.image(image, caption='DeepFM architecture')

with tab4:
    st.write("Deep Cross Network (DCN) have two parallel networks: a cross network and a deep network. Cross network is used to depict high-degree and explicit feature interactions and deep network is used to capture implicit feature interactions.")
    image = Image.open('figs/DCN.png')
    st.image(image, caption='DCN architecture')


with tab5:
    st.write("Deep Cross Network v2 (DCNv2) applies a matrix to calculate the weight of feature interactions while the DCN uses a vector weigh.")
    image = Image.open('figs/cross_dcn.png')
    st.image(image, caption='The cross layer in DCN')
    image = Image.open('figs/cross_dcnv2.png')
    st.image(image, caption='The cross layer in DCNv2')

with tab6:
    st.write("Attention Factorization Machine (AFM) uses an attention network to learn the attention for different feature interactions.")
    image = Image.open('figs/AFM.png')
    st.image(image, caption='AFM architecture')

st.header('3. Experiment Results')
st.write("Test AUC:")

methods = ['FM' ,'IPNN' ,'DeepFM','DCN','DCNv2','AFM']
auc_criteo = [0.7676 ,0.7860 ,0.7852 ,0.7854 ,0.7858 ,0.7744]
auc_avazu = [0.7564 ,0.7573 ,0.7566 ,0.7562 ,0.7573 ,0.7560]

df = pd.DataFrame(
   np.array([auc_avazu,auc_criteo]),
            index=['Avazu','Criteo'],
            columns=methods)
st.dataframe(df)

st.subheader('Effects of Embedding Size')
tab1, tab2 = st.tabs(['Avazu','Criteo'])
with tab1:
    st.write("Best performance: embedding size 40.")
    image = Image.open('figs/emb_avazu.png')
    st.image(image)

with tab2:
    st.write("Best performance: embedding size 20.")
    image = Image.open('figs/emb_criteo.png')
    st.image(image)


