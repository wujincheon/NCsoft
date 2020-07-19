import pickle
import numpy as np
import numpy
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm
import pandas as pd
import csv

SEQUENCE_LENGTH = 9 #패턴을 찾아낼 시퀀스의 길이 설정
EMBEDDING_DIM = 100 # 로그아이디 임베딩 사이즈
HIDDEN_SIZE = 100
KEEP_PROB = 0.5
BATCH_SIZE = 512
class_num=127 #
MODEL_PATH = './model'


def gen_seqs(data):  # 유저별 로그아이디 시퀀스로 되어있는 데이터를 길이 10의 시퀀스로 잘라서 학습, 테스트 데이터로 분할 // 그리고 각 로그아이디를 0부터 enumerate시켜서 변환해주는 함수

    sentences = []
    for i in data['sequence']:
        sentence = []

        for j in range(len(i)):
            sentence.append(i[j][0])
        sentences.append(sentence)

    ### 길이 10보다 작으면 그대로, 10보다 크면 10짜리 사이즈로 sliding하면서 생성
    final_sen = []
    for i in sentences:
        if len(i) < 10:
            final_sen.append(i)
        else:
            for j in range(len(i) - 10):
                final_sen.append(i[j:j + 10])

    ### 100만개를 여러개 샘플링하며 실험한 결과 아래와 같이 분할하여 진행 , 후에 test_seqs에 대해서 패턴을 뽑아낼 예정
    train_seqs = final_sen[2000000:2700000]
    test_seqs = final_sen[:2000000] + final_sen[2700000:]

    # 각 로그 아이디를 0부터 순서대로 할당해주는 딕셔너리 생성
    a = set()
    for i in final_sen:
        a = a.union(set(i))
    item2index = {}
    for item in a:
        item2index[item] = len(item2index)

    index2item = {str(j): i for i, j in item2index.items()}  # 다시 아이템으로 표시하기 위한 딕셔너리를 생성해둠

    # 딕셔너리를 이용해 위의 분할된 시퀀스의 로그아이디들을 인덱스로 변환
    for i in range(len(train_seqs)):
        for j in range(len(train_seqs[i])):
            train_seqs[i][j] = item2index[train_seqs[i][j]]
    for i in range(len(test_seqs)):
        for j in range(len(test_seqs[i])):
            test_seqs[i][j] = item2index[test_seqs[i][j]]

    return train_seqs, test_seqs, index2item


def process_seqs(iseqs):  # 위에서 구한 sequence에 대해 , 시작 위치부터 1씩 sliding 하면서 training data를 augmentation하는 함수
    out_seqs = []

    labs = []
    for seq in iseqs:
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]

    return (out_seqs, labs)


def prepare_data(seqs, labels):  # 길이가 10이하인 시퀀스도 있기에, 나머지 자리에 zero-padding을 해주는 함수

    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    # maxlen=148
    x = numpy.zeros((n_samples, maxlen)).astype('int64')
    x_mask = numpy.ones((n_samples, maxlen)).astype(np.float64)
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s

    x_mask *= (1 - (x == 0))

    return x, x_mask, labels


def batch_generator(X, y, batch_size): #batch data 만드는 함수
    """Primitive batch generator
    """
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


# 패턴에 대한 정보인 patterns를 이용해 패턴의 길이를 지정하여 각 패턴과 그 softmax 확률 누적값에 대한 정보를 추출하는 함수
def patterns_len_k(patterns, k, index2item): # index2item은 인덱스로 나타낸 아이템들을 다시 원래의 로그 아이디로 나타내기 위한 딕셔너리임
    pattern_len_k_keys = []
    for i in patterns.keys():
        if (i.count(',') + 1) == k:
            pattern_len_k_keys.append(i)

    print("길이 %d의 패턴 개수 : %d개" % (k, len(pattern_len_k_keys)))
    pattern_list = {i: patterns[i] for i in pattern_len_k_keys}
    pattern_list = sorted(pattern_list.items(), key=lambda x: x[1], reverse=True)

    final_pattern_list = {}
    for i in pattern_list:
        log_id = []
        for j in i[0][1:-1].split(', '):
            log_id.append(index2item[j[1:-1]])

        final_pattern_list[str(log_id)] = i[1]

    final_pattern_list = sorted(final_pattern_list.items(), key=lambda x: x[1], reverse=True)
    return final_pattern_list

########### 모델 구현 ##########

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None, 127], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([class_num, EMBEDDING_DIM], -1.0, 1.0), trainable=True) #shape 가 vocab size , emb dim -1 ~ 1 사이의
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                        inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
tf.summary.histogram('RNN_outputs', rnn_outputs)
drop_input= tf.concat(rnn_outputs, 2)[:,8,:] # 마지막 hidden_state 를 이용
# Dropout
drop = tf.nn.dropout(drop_input, keep_prob_ph)

drop_list=[] # test시 각 시점에서의 hidden_state가 final hidden state이기 때문에 모든 시점의 hidden state를 구해놓음
for i in range(0,8):
    drop_list.append(tf.nn.dropout(tf.concat(rnn_outputs, 2)[:,i,:], keep_prob_ph))

# Fully connected layer # softmax 2 까지!!
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(
        tf.truncated_normal([HIDDEN_SIZE * 2, 127], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[127]))
    y_hat = tf.nn.softmax(tf.nn.xw_plus_b(drop, W, b))
    tf.summary.histogram('W', W)

    ### 각 시점별로 top2의 인덱스와 softmax값을 얻어내는 과정,, concat을 이용해서 합치기 위해 첫 값만 for문을 쓰기에 앞서 정의함
    y_hat0 = tf.nn.softmax(tf.nn.xw_plus_b(drop_list[0], W, b))
    value0, y_hat0 = tf.nn.top_k(y_hat0, 2)

    y_hat0_index1 = tf.expand_dims(y_hat0[:, 0], 1)  # 두개까지 허용하기로 하여 top_2의 인덱스값을 저장
    y_hat0_index2 = tf.expand_dims(y_hat0[:, 1], 1)
    y_hat0_softmax1 = tf.expand_dims(value0[:, 0], 1)  # 두개까지 허용하기로 하여 top_2의 softmax값을 저장
    y_hat0_softmax2 = tf.expand_dims(value0[:, 1], 1)

    for i in range(1, 8):
        y_hati = tf.nn.softmax(tf.nn.xw_plus_b(drop_list[i], W, b))
        valuei, y_hati = tf.nn.top_k(y_hati, 2)

        y_hati_index1 = tf.expand_dims(y_hati[:, 0], 1)# 두개까지 허용하기로 하여 top_2의 인덱스값을 저장
        y_hati_index2 = tf.expand_dims(y_hati[:, 1], 1)
        y_hati_softmax1 = tf.expand_dims(valuei[:, 0], 1)# 두개까지 허용하기로 하여 top_2의 softmax값을 저장
        y_hati_softmax2= tf.expand_dims(valuei[:, 1], 1)

        y_hat_total_index1 = tf.concat([y_hat0_index1, y_hati_index1], 1)  # 각 시점에서의 예측 스코어에 대한 top2 인덱스값들을 저장
        y_hat_total_index2 = tf.concat([y_hat0_index2, y_hati_index2], 1)

        y_hat_total_softmax1 = tf.concat([y_hat0_softmax1, y_hati_softmax1], 1)  # 각 시점에서의 예측 스코어에 대한 top2 softmax값들을 저장
        y_hat_total_softmax2 = tf.concat([y_hat0_softmax2, y_hati_softmax2], 1)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    ##기존 로스방식

    loss = tf.reduce_mean(-tf.reduce_sum(target_ph * tf.log(y_hat), reduction_indices=1))
    tf.summary.scalar('loss', loss)
    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # recall   #모델이 예측한 top2 안에 ground truth가 있으면 hit로 하여 계산
    ranks1 = tf.reduce_sum(target_ph * y_hat, reduction_indices=1)
    ranks2 = tf.cast(y_hat >= tf.tile(tf.expand_dims(ranks1, 1), [1, 127]), tf.float32)
    ranks = tf.reduce_sum(ranks2, reduction_indices=1)
    rank_ok = tf.cast((ranks <= 2), tf.float32)
    recall = tf.reduce_mean(rank_ok)
    tf.summary.scalar('recall', recall)

merged = tf.summary.merge_all()

if __name__ == "__main__":

    with open("../01_DATA/seq_data_LogId_sample_10000_1.p", mode='rb') as f:  # actor id 10000개에 대한 데이터 이용
        data = pickle.load(f)

    train_seqs, test_seqs, index2item = gen_seqs(data)
    train = process_seqs(train_seqs)
    test = process_seqs(test_seqs)

    train_x, mask, train_y = prepare_data(train[0], train[1])
    test_x, mask2, test_y = prepare_data(test[0], test[1])
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    # Batch generators 생성
    train_batch_generator = batch_generator(train_x, train_y, BATCH_SIZE)
    test_batch_generator = batch_generator(test_x, test_y, BATCH_SIZE)

    train_writer = tf.summary.FileWriter('./logdir/train', recall.graph)
    test_writer = tf.summary.FileWriter('./logdir/test', recall.graph)

    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    saver = tf.train.Saver()

    with tf.Session(config=session_conf) as sess:
        train_writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        patterns = {}
        for epoch in range(30):
            loss_train = 0
            loss_test = 0
            recall_train = 0
            recall_test = 0
            mrr_train = 0
            mrr_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = train_x.shape[0] // BATCH_SIZE
            # num_batches=1
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)
                y_batch_list = []
                for i in y_batch: # softmax의 형태로 만들어 주기 위해 타겟을 변형
                    a = [0] * 127
                    a[i - 1] = 1
                    y_batch_list.append(a)
                y_batch2 = np.array(y_batch_list)
                seq_len = np.array(
                    [list(x).index(0) + 1 if list(x)[-1] == 0 else 9 for x in x_batch])  # actual lengths of sequences, 제로패딩을 감안함
                loss_tr, reca, _, summary = sess.run([loss, recall, optimizer, merged],
                                                     feed_dict={batch_ph: x_batch,
                                                                target_ph: y_batch2,
                                                                seq_len_ph: seq_len,
                                                                keep_prob_ph: KEEP_PROB})
                recall_train = recall_train + reca
                loss_train = loss_tr
                train_writer.add_summary(summary, b + num_batches * epoch)
            recall_train /= num_batches
            loss_train /= num_batches
            mrr_train /= num_batches

            # Testing
            num_batches = test_x.shape[0] // BATCH_SIZE  # 전체 테스트데이터에 대해 패턴만 뽑아내기 위해 이렇게 설정함

            num_batches2 = 0  # 이미 test_loss 검증이 끝났기에 0으로 넣어놓았으나, 검증이 다시 필요할경우 아래의 배치수를 num_batches2 -> num_batches로 변경하면 됨
            for b in tqdm(range(num_batches2)):
                x_batch, y_batch = next(test_batch_generator)
                y_batch_list = []
                for i in y_batch:
                    a = [0] * 127
                    a[i - 1] = 1
                    y_batch_list.append(a)
                y_batch2 = np.array(y_batch_list)
                seq_len = np.array(
                    [list(x).index(0) + 1 if list(x)[-1] == 0 else 9 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, reca, summary = sess.run([loss, recall, merged],
                                                          feed_dict={batch_ph: x_batch,
                                                                     target_ph: y_batch2,
                                                                     seq_len_ph: seq_len,
                                                                     keep_prob_ph: 1.0})

                recall_test = recall_test + reca
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            recall_test /= num_batches
            loss_test /= num_batches
            mrr_test /= num_batches

            print(
                "loss_train: {:.3f}, loss_test: {:.3f}, recall_train: {:.10f}, recall_test: {:.10f}, mrr_train: {:.3f}, mrr_test: {:.3f}".format(
                    loss_train, loss_test, recall_train, recall_test, mrr_train, mrr_test
                ))

        # 패턴을 쌓기위한 과정
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(test_batch_generator)
            y_batch_list = []
            for i in y_batch:
                a = [0] * 127
                a[i - 1] = 1
                y_batch_list.append(a)
            y_batch2 = np.array(y_batch_list)
            seq_len = np.array(
                [list(x).index(0) + 1 if list(x)[-1] == 0 else 9 for x in x_batch])  # actual lengths of sequences
            loss_test_batch, reca, pred_1, pred_2, softmax1, softmax2, summary = sess.run(
                [loss, recall, y_hat_total_index1, y_hat_total_index2, y_hat_total_softmax1, y_hat_total_softmax2, merged],
                feed_dict={batch_ph: x_batch,
                           target_ph: y_batch2,
                           seq_len_ph: seq_len,
                           keep_prob_ph: 1.0})



            # 각 시점에서 예측한 아이템 top2를 각각 pred_1,2로 지정한 뒤 g.t 값과 비교한 boolean matrix를 만들어서 어느지점에서 cut할지 결정하는 cut_list를 만든다.
            cut_list1 = (pred_1 == x_batch[:, 1:])
            cut_list2 = (pred_2 == x_batch[:, 1:])

            cut_list = cut_list1 + cut_list2
            pattern = []
            softmax_value = 0
            # 위의 cut list를 기준으로 패턴을 쌓는 코드 # cut_list 가 True이면 패턴으로 쌓고, 그게 몇번째 index인지에 따라 해당 softmax값을 누적해서 저장하는 코드
            for k in range(len(cut_list)):
                for i in range(len(cut_list[k])):
                    if cut_list[k][i] == False:
                        if len(pattern) > 1:
                            if pattern[0][0] != 0:
                                key = str(pattern)
                                if key not in patterns.keys():
                                    patterns[key] = 0
                                patterns[key] += softmax_value
                            pattern = []
                            softmax_value = 0

                    if cut_list[k][i] == True:
                        if len(pattern) == 0:
                            pattern.append(list(x_batch[0][i:i + 1]))
                            pattern.append(list(x_batch[0][i + 1:i + 2]))
                            if cut_list1[k][i] == True:
                                softmax_value += softmax1[k][i]
                            if cut_list2[k][i] == True:
                                softmax_value += softmax2[k][i]

                        else:
                            pattern.append(list(x_batch[0][i + 1:i + 2]))
                            if cut_list1[k][i] == True:
                                softmax_value += softmax1[k][i]
                            if cut_list2[k][i] == True:
                                softmax_value += softmax2[k][i]
                    if i == 7:
                        if len(pattern) != 0:
                            if pattern[0][0] != 0:
                                key = str(pattern)
                                if key not in patterns.keys():
                                    patterns[key] = 0
                                patterns[key] += softmax_value
                            pattern = []
                            softmax_value = 0

        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")


    # 길이 4,5,6 에 해당하는 패턴들을 누적 softmax값을 기준으로 원하는 만큼(100개) 저장하는 함수
    with open('pattern_len4.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for i in patterns_len_k(patterns, 4, index2item)[:100]:
            writer.writerow(i[0][2:-2].split("', '"))
    csvFile.close()

    import csv

    with open('pattern_len5.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for i in patterns_len_k(patterns, 5, index2item)[:100]:
            writer.writerow(i[0][2:-2].split("', '"))
    csvFile.close()

    import csv

    with open('pattern_len6.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for i in patterns_len_k(patterns, 6, index2item)[:100]:
            writer.writerow(i[0][2:-2].split("', '"))
    csvFile.close()