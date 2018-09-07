import tensorflow as tf
import os
from models.tensorflow_fm.tensorflow_fm import MyModel
from models.dataset_fn.ctr_dataset import get_feature_dim, get_input_fn

tf.app.flags.DEFINE_integer("thresh_hold", 0, 'thresh_hold')
FLAGS = tf.app.flags.FLAGS


def main(_):
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(curr_dir, 'logdir')
    train_path = os.path.join(curr_dir, '../../preprocess/output/train_shuffle_' + str(FLAGS.thresh_hold) + '.csv')
    valid_path = os.path.join(curr_dir, '../../preprocess/output/valid_raw_' + str(FLAGS.thresh_hold) + '.csv')

    batch_size = 2048
    params = {
        'alpha': 0,
        'feature_dim': get_feature_dim(FLAGS.thresh_hold),
        'hidden_fields': 3,
        'batch_size': batch_size
    }
    model = MyModel(model_dir=model_dir, params=params)
    train_input_fn = get_input_fn(train_path, batch_size, 1, 10000)
    valid_input_fn = get_input_fn(valid_path, batch_size, 1, 10000)
    model.train(input_fn=train_input_fn)
    model.evaluate(input_fn=valid_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
