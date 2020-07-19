import tensorflow as tf
import numpy as np
from numpy import genfromtxt

import sklearn
from sklearn.preprocessing import StandardScaler, scale
from sklearn import (datasets, random_projection)
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

import csv
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame

from datetime import datetime

from time import time

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance



tf.set_random_seed(777)  # reproducibility

'''피처 스케일링을 위한 함수'''
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)



'''raw data를 읽고 판다스의 데이터 프레임 형식으로 저장하여 minmax 스케일링을 진행'''
def data_load(path):
    data=genfromtxt(path, delimiter=',') #
    data=data[1:,1:]
    df=DataFrame(data)
    dataset_temp = df.as_matrix()
    dataset = MinMaxScaler(dataset_temp)

    return dataset


# train Parameters
'''25일치의 연속된 정보를 반영한 하루를 나타내기 위해 25일씩 학습, 그 외의 히든 레이어의 dim이나 학습률, stack등은 실험을 통해 얻은 값으로, global optimum은 아닐 수 있음'''


'''25일의 배치사이즈를 위해 변수들 x를 만들어내고, 예측하고자 하는 다음날의 값을 y로 저장'''
def data_generate(dataset):
    dataX = []
    dataY = []
    for i in range(0, len(dataset) - seq_length - predict_day + 1): # 몇일 뒤의 값을 타겟으로 할지 predict_day를 미리 설정, 본 과제에서는 하루로 설정.
        _x = dataset[i:i + seq_length]
        _y = dataset[i + predict_day:i + seq_length + predict_day]
        # print(i + seq_length+predict_day)
        dataX.append(_x)
        dataY.append(_y)

        x = np.array(dataX)
        y = np.array(dataY)
        y_label = y[:, :, :]

    return x, y_label

'''tensor의 인풋으로 사용할 array 형식으로 만들어주며, 학습시 label로 사용할 array를 따로 생성'''


# build a LSTM network
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    return cell

def train_representation(x, y_label, seq_length, data_dim, hidden_dim, output_dim,learning_rate,iterations,LSTM_stack):

    # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name='intput_X')
    Y = tf.placeholder(tf.float32, [None, seq_length, data_dim], name='intput_Y')
    multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(LSTM_stack)], state_is_tuple=True)
    outputs_rnn, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
    X_for_fc = tf.reshape(outputs_rnn, [-1, hidden_dim])
    Y_pred_temp = tf.contrib.layers.fully_connected(X_for_fc, output_dim, activation_fn=None)
    # reshape out for sequence_loss
    Y_pred = tf.reshape(Y_pred_temp, [-1, seq_length, output_dim])

    # cost/loss
    mean_loss = tf.reduce_mean(tf.square(Y_pred - Y), name='losses_mean')  # mean of the squares
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)


    '''tensorflow session 생성 및 학습'''
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    losslist= [] #loss가 잘 줄어드는지, 학습이 잘되는지 파악하고 하는 용도

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([optimizer, mean_loss],
                                feed_dict={X: x,
                                           Y: y_label})
        if i%50 ==0:
            print("%d %% 진행중" % (int(i) / 5))
        losslist = np.append(losslist, step_loss) # 수렴하는지 따로 그래프를 그려볼 수 있음

    '''학습이 끝난 모델에 대해서 1~25일을 반영한 26일치의 representation을 얻어내는 과정'''
    representation = sess.run([Y_pred], feed_dict={X: x[0].reshape([1, 25, 385]), Y: y_label[0].reshape([1, 25, 385])})
    representations = representation[0][0][24]

    '''representations 에 전체 날, 26~3230 일짜의 representation을 차례대로 추가해줌'''
    for i in range(1, 3205):
        representation = sess.run([Y_pred],
                                  feed_dict={X: x[i].reshape([1, 25, 385]), Y: y_label[i].reshape([1, 25, 385])})
        representations = np.vstack((representations, representation[0][0][24]))

    return representations

'''t-SNE로 임베딩한 representation을 2차원상에서 매핑시켜주는 함수입니다'''
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(100, 100))
    ax = plt.subplot(111)
    for i in range(X.shape[0]): # 1년이 지날때마다 색을 바꿔서 시간의 흐름에 얼마나 뭉쳐있는지 확인하기 위함
        plt.text(X[i, 0], X[i, 1], str(y_date[i]),
                 color=plt.cm.Set1(y_date[i] // 365),
                 fontdict={'weight': 'light', 'size': 70})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(3205):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)

def clust(X):
    '''(시행착오로 결정된) threshold 0.05 거리를 기준으로 군집화하는 코드입니다.
    먼저 순차적으로 모든날에 대해 그 날과 근접한 날들을 모두 찾습니다.'''
    nearest_list = [[i] for i in range(26, 3231)]
    for j in range(3205):
        for k in range(j + 1, 3205):
            if distance.euclidean(X[j], X[k]) < 0.05:
                nearest_list[j].append(k + 26)

    '''각 날들에 대해 근접한 날을 묶었으니, 겹치지 않도록 합쳐주는 작업을 진행합니다'''
    cluster_list = [[] for k in range(1300)]  # 군집들을 저장할 리스트,, 나중에 이를 출력하여 시퀀스 차이가 큰데 같은 군집에 있는 날짜들에 대한 기사를 수집할 예정.
    l = 0
    for i in range(3205):
        a = 0
        for j in range(l + 1):
            if len(set(nearest_list[i]).intersection(cluster_list[j])) != 0:
                cluster_list[j] = list(set(nearest_list[i]) | set(cluster_list[j]))
                break
            else:
                a = a + 1
        if a > l:
            cluster_list[l] = nearest_list[i]
            l = l + 1
    return cluster_list

if __name__ == "__main__":
    seq_length = 25
    data_dim = 385
    hidden_dim = 70
    output_dim = 385
    predict_day = 1
    learning_rate = 0.01
    iterations = 500
    LSTM_stack = 2

    dataset= data_load('../01_DATA/05_RNN_embedding/Total_index.csv')

    x, y_label = data_generate(dataset)


    '''26일부터 3230일까지를 숫자로 label하여 그래프에 나타내기 위함'''
    y1 = list(range(26, 3231))
    y_date = np.array(y1)

    representations = train_representation(x, y_label, seq_length, data_dim, hidden_dim, output_dim,learning_rate,iterations,LSTM_stack)


    '''앞서 만들어 놓은 임베딩 벡터를 X로 지정하고, scikit learn의 t-SNE함수에 적용시켜 2차원으로 매핑한 후, 위의 plot_embedding 함수를 이용해서 시각화까지 진행한 함수입니다.'''
    embedding = TSNE(learning_rate=10)
    transformed = embedding.fit_transform(representations)


    t0=time()
    plot_embedding(transformed,"t_SNE projection of the digits (time %.2fs)" %(time() - t0)) #365일씩

    #임베딩한 공간에서 일정 거리를 기준으로 군집화를 합니다.
    print(clust(representations))

    '''군집들이 저장된 m을 출력하여, 군집내에 30일 이상 차이가 나는 시퀀스의 경우, 두 시작 날을 저장 (총 6쌍과 이후 200일 단위로 12개 날짜 선택하여 총 24개의 날짜를 선정) -> 이를 이용해 기사를 수집하여 단어 분포의 유사도를 비교할 예정'''
    '''many2many_visualization : 각 날들의 representation 을 2차원으로 임베딩한 그래프
       cluster_list : 일정 거리 이내를 기준으로 근접한 임베딩을 가지는 날들의 군집'''
