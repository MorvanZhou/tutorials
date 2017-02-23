# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# Save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
# if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#     init = tf.initialize_all_variables()
# else:
#     init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, "my_net/save_net.ckpt")
#    print("Save to path: ", save_path)


################################################
# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))















