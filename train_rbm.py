import tensorflow.compat.v1 as tf
from dataset import _get_training_data, _get_test_data
from rbm_model import RBM
import numpy as np
import sys
sys.argv = sys.argv[:1]
tf.compat.v1.disable_eager_execution()


tf.app.flags.DEFINE_string('tf_records_train_path', '/Users/phusdt/GitHub/Recommender-System/tf_records_100k/train/',
                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_test_path', '/Users/phusdt/GitHub/Recommender-System/tf_records_100k/test/',
                           'Path of the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',1.0,
                          'Learning_Rate')

tf.app.flags.DEFINE_integer('num_v', 1682,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 10,
                            'Number of hidden neurons.')

tf.app.flags.DEFINE_integer('num_samples', 943,
                            'Number of training samples (Number of users, who gave a rating).')

tf.app.flags.DEFINE_integer('k', 10,
                           'Number of iterations during gibbs samling.')

tf.app.flags.DEFINE_integer('eval_after',50,
                            'Evaluate model after number of iterations.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    '''Building the graph, opening of a session and starting the training od the neural network.'''
    
    num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

    with tf.Graph().as_default():

        train_data, train_data_infer=_get_training_data(FLAGS)
        test_data=_get_test_data(FLAGS)

        iter_train = tf.data.make_initializable_iterator(train_data)
        iter_train_infer = tf.data.make_initializable_iterator(train_data_infer)
        iter_test = tf.data.make_initializable_iterator(test_data)

        x_train= iter_train.get_next()
        x_train_infer=iter_train_infer.get_next()
        x_test=iter_test.get_next()

        model=RBM(FLAGS)

        update_op, accuracy=model.optimize(x_train)
        v_infer=model.inference(x_train_infer)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.num_epoch):

                acc_train=0
                acc_infer=0

                sess.run(iter_train.initializer)

                for batch_nr in range(num_batches):
                    _, acc=sess.run((update_op, accuracy))
                    acc_train+=acc

                    if batch_nr>0 and batch_nr%FLAGS.eval_after==0:

                        sess.run(iter_train_infer.initializer)
                        sess.run(iter_test.initializer)

                        num_valid_batches=0

                        for i in range(FLAGS.num_samples):

                            v_target=sess.run(x_test)[0]

                            if len(v_target[v_target>=0])>0:

                                v_=sess.run(v_infer)[0]
                                acc=1.0-np.mean(np.abs(v_[v_target>=0]-v_target[v_target>=0]))
                                acc_infer+=acc
                                num_valid_batches+=1

                        print('epoch_nr: %i, batch: %i/%i, acc_train: %.3f, acc_test: %.3f'%
                            (epoch, batch_nr, num_batches, (acc_train/FLAGS.eval_after), (acc_infer/num_valid_batches)))

                        acc_train=0
                        acc_infer=0 
     
if __name__ == "__main__":
    
    tf.app.run()