import tensorflow as tf

# ======建立模型(開始)======

# 建立權重張量的函數
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

# 建立偏差張量的函數
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

# 建立卷積層的函數
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 建立池化層的函數
def max_pool_2x2(x):
    return tf.nn.max_pool(x, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')


# inference 是推論的意思
def inference(images, batchSize, nClasses):
    # 建立輸入層
    # x_input = tf.placeholder(shape=[None, 160, 160, 3], dtype=tf.float32, name='x')

    # 建立卷積層 1
    W1 = weight([4, 4, 3, 30])
    b1 = bias([30])
    C1_Conv = tf.nn.relu(conv2d(images, W1) + b1)

    # 建立池化層 1
    C1_Pool = max_pool_2x2(C1_Conv)

    # 建立卷積層 2
    W2 = weight([4, 4, 30, 60])
    b2 = bias([60])
    C2_Conv = tf.nn.relu(conv2d(C1_Pool, W2) + b2)

    # 建立池化層 2
    C2_Pool = max_pool_2x2(C2_Conv)

    # 建立卷積層 3
    W3 = weight([4, 4, 60, 90])
    b3 = bias([90])
    C3_Conv = tf.nn.relu(conv2d(C2_Pool, W3) + b3)

    # 建立池化層 3
    C3_Pool = max_pool_2x2(C3_Conv)

    # 建立卷積層 4
    W4 = weight([4, 4, 90, 120])
    b4 = bias([120])
    C4_Conv = tf.nn.relu(conv2d(C3_Pool, W4) + b4)

    # 建立池化層 4
    C4_Pool = max_pool_2x2(C4_Conv)

    # 建立平坦層
    D_Flat = tf.reshape(C4_Pool, shape=[batchSize, -1])
    dim = D_Flat.shape[1].value

    # 建立隱藏層 1
    W5 = weight([dim, 300])
    b5 = bias([300])
    D_Hidden1 = tf.nn.relu(tf.matmul(D_Flat, W5) + b5)
    D_Hidden1_Dropout = tf.nn.dropout(D_Hidden1, keep_prob=1)

    # 建立隱藏層 2
    W6 = weight([300, 250])
    b6 = bias([250])
    D_Hidden2 = tf.nn.relu(tf.matmul(D_Hidden1_Dropout, W6) + b6)
    D_Hidden2_Dropout = tf.nn.dropout(D_Hidden2, keep_prob=1)

    # 建立輸出層
    W7 = weight([250, nClasses])
    b7 = bias([nClasses])
    y_predict = tf.add(tf.matmul(D_Hidden2_Dropout, W7), b7, name='y_output')

    return y_predict

# ======建立模型(結束)======



# ======定義誤差函數(開始)======
def losses(y_predict, y_label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    return loss
# ======定義誤差函數(結束)======



# ========定義訓練方式(開始)========
def trainning(loss, learning_rate):
    # 定義優化方法
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer
# ========定義訓練方式(結束)========



# ========定義評估模型準確率的方法(開始)========
def evaluation(y_predict, y_label):
    # 拿預測結果和標籤做比較，計算是否判斷正確
    correct_prediction = tf.equal(tf.arg_max(y_predict, 1),
                                  tf.arg_max(y_label, 1))

    # 計算預測正確結果的平均，計算預測準確率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    return accuracy
# ========定義評估模型準確率的方法(結束)========