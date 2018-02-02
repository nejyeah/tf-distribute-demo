# encoding:utf-8
import tempfile
import time
import tensorflow as tf
import numpy as np
import model

flags = tf.app.flags
flags.DEFINE_integer('image_pixels', 28, 'size of a squre image')
flags.DEFINE_integer('num_classes', 10, 'num of classes')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '10.240.208.65:8888', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '10.240.209.91:8888,10.240.209.91:9999',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS

def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print 'job_name : %s' % FLAGS.job_name
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print 'task_index : %d' % FLAGS.task_index

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        x = tf.placeholder(tf.float32, [None, FLAGS.image_pixels * FLAGS.image_pixels])
        y_ = tf.placeholder(tf.int32, [None])

        logits = model.mlp(x, FLAGS.image_pixels, FLAGS.hidden_units, FLAGS.num_classes) 
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        # accuracy
        accuracy = tf.metrics.accuracy(y_, tf.argmax(logits, axis=1))

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)

        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        # Supervisor is deprecated, using "MonitoredTrainingSession" instead
        # logdir: checkpoint save dir
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1, global_step=global_step)

        if is_chief:
            print 'Worker %d: Initailizing session...' % FLAGS.task_index
        else:
            print 'Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index

        sess = sv.prepare_or_wait_for_session(server.target)
        print 'Worker %d: Session initialization  complete.' % FLAGS.task_index

        time_begin = time.time()
        print 'Traing begins @ %f' % time_begin

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_op, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print '%f: Worker %d: traing step %d dome (global step:%d)' % (now, FLAGS.task_index, local_step, step)

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print 'Training ends @ %f' % time_end
        train_time = time_end - time_begin
        print 'Training elapsed time:%f s' % train_time

        if is_chief:
          val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
          val_xent = sess.run(cross_entropy, feed_dict=val_feed)
          val_accuracy = sess.run(accuracy, feed_dict=val_feed)
          print 'After %d training step(s)' % (FLAGS.train_steps)
          print('cross_entropy:')
          print(np.mean(val_xent))
          print("accuracy:")
          print(val_accuracy)
        sess.close()

if __name__ == '__main__':
    tf.app.run()
