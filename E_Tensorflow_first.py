import tensorflow as tf
import A_tfrecords
'''
    作者：石高辉
    功能：将人脸图像进行二分类
'''


def forward(input, w, b, keepratio):
    input_r = tf.reshape(input, shape=[-1, 40, 40, 1])  # 输入四维向量
    # Layer1
    conv1 = tf.nn.conv2d(input_r, w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b['bc1']))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout1 = tf.nn.dropout(pool1, keepratio)
    # Layer2
    conv2 = tf.nn.conv2d(dropout1, w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b['bc2']))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout2 = tf.nn.dropout(pool2, keepratio)
    # Layer3
    dense = tf.reshape(dropout2, [-1, w['wf1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(dense, w['wf1']), b['bf1'])
    relu3 = tf.nn.relu(fc1)
    dropout3 = tf.nn.dropout(relu3, keepratio)
    # Layer4
    fc2 = tf.add(tf.matmul(dropout3, w['wf2']), b['bf2'])
    dropout4 = tf.nn.dropout(fc2, keepratio)
    # Layer5
    out = tf.add(tf.matmul(dropout4, w['wf3']), b['bf3'])

    output = {
        'input_r': input_r, 'conv1': conv1, 'relu1': relu1, 'pool1': pool1, 'dropout1': dropout1,
        'conv2': conv2, 'relu2': relu2, 'pool2': pool2, 'dropout2': dropout2,
        'dense': dense, 'fc1': fc1, 'relu3': relu3, 'dropout3': dropout3,
        'fc2': fc2, 'dropout4': dropout4,
        'out': out
    }
    return output


if __name__=="__main__":
    n_input = 40*40*1     #输入图像为28*28*3
    n_classes = 2

    #input and output
    input = tf.placeholder(tf.float32,[None,n_input])
    output = tf.placeholder(tf.float32,[None,n_classes])
    keepratio = tf.placeholder(tf.float32)
    #network param
    stddev = 0.1
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64],stddev=stddev)),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128],stddev=stddev)),
        'wf1': tf.Variable(tf.random_normal([10*10*128, 1024],stddev=stddev)),
        'wf2': tf.Variable(tf.random_normal([1024, 1024],stddev=stddev)),
        'wf3': tf.Variable(tf.random_normal([1024, n_classes],stddev=stddev))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64],stddev=stddev)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
        'bf1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
        'bf2': tf.Variable(tf.random_normal([1024], stddev=stddev)),
        'bf3': tf.Variable(tf.random_normal([n_classes], stddev=stddev))
    }

    #损失函数和优化
    pred = forward(input, weights, biases, keepratio)['out'] #预测值
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output,logits=pred))
    optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    corr = tf.equal(tf.argmax(pred,1),tf.argmax(output,1))
    accr = tf.reduce_mean(tf.cast(corr, tf.float32))
    init = tf.global_variables_initializer()

    #保存模型
    save_step = 10
    saver = tf.train.Saver(max_to_keep=30)
    do_train = 1

    #计算
    train_epochs = 3000
    batch_size = 6000
    display_step = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        if do_train == 1:  # 训练
            ckpt = tf.train.get_checkpoint_state("tensorflow_first")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                for epoch in range(211, train_epochs):
                    avg_cost = 0
                    num_batch = int(6000 / batch_size)  # train_set_size / batch_size
                    for i in range(num_batch):  # 每个batch的送入，大小为batch_size * pic_size(500 * 28*28*1)
                        img_batch, label_batch = A_tfrecords.get_tfrecord(batch_size, isTrain=False)
                        coord = tf.train.Coordinator()
                        thread = tf.train.start_queue_runners(sess, coord)
                        data, label = sess.run([img_batch, label_batch])
                        sess.run(optm, feed_dict={input: data, output: label, keepratio: 0.7})
                        feeds = {input: data, output: label, keepratio: 1.}
                        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
                    # 显示
                    if epoch % display_step == 0:
                        print("Epoch: %04d/%04d cost: %.4f" % (epoch, train_epochs, avg_cost))
                        train_acc = sess.run(accr, feed_dict={input: data, output: label, keepratio: 1})
                        print("Training Accuracy: %.3f" % (train_acc))
                    # 保存模型
                    if epoch % save_step == 0:
                        saver.save(sess, "tensorflow_first/cnn_basic.ckpt-" + str(epoch))
                print("Train Finished!")

        if do_train == 0:
            img_batch, label_batch = A_tfrecords.get_tfrecord(6000, isTrain=False)
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess, coord)
            data, label = sess.run([img_batch, label_batch])
            epoch = train_epochs - 10
            saver.restore(sess, "tensorflow_first/cnn_basic.ckpt-" + str(200))
            test_acc = sess.run(accr, feed_dict={input: data, output: label, keepratio: 1})
            print("Test Accuracy: %.3f" % (test_acc))
