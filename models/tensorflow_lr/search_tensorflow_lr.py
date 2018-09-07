import tensorflow as tf
import os
from models.tensorflow_lr.tensorflow_lr import MyModel
from models.dataset_fn.ctr_dataset import get_feature_dim, get_input_fn
from advisor_client.client import AdvisorClient

tf.app.flags.DEFINE_integer("thresh_hold", 0, 'thresh_hold')
FLAGS = tf.app.flags.FLAGS


def main(_):
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    train_path = os.path.join(curr_dir, '../../preprocess/output/train_shuffle_' + str(FLAGS.thresh_hold) + '.csv')
    valid_path = os.path.join(curr_dir, '../../preprocess/output/valid_raw_' + str(FLAGS.thresh_hold) + '.csv')

    warm_start_dir = os.path.join(curr_dir, 'logdir')
    train_input_fn = get_input_fn(train_path, 4096, 1, 10000)
    valid_input_fn = get_input_fn(valid_path, 4096, 1, 10000)

    client = AdvisorClient()

    alpha_list = [0.00001, 0.0001, 0.001, 0.01]
    params = {
        'alpha': 0,
        'feature_dim': get_feature_dim(FLAGS.thresh_hold)
    }
    # ws = tf.estimator.WarmStartSettings(
    #     ckpt_to_initialize_from=warm_start_dir,
    #     vars_to_warm_start=".*input_layer.*")
    ws = None
    for alpha in alpha_list:
        params['alpha'] = alpha
        model_dir = os.path.join(curr_dir, 'logdir_s' + str(alpha))
        model = MyModel(model_dir=model_dir, params=params, warm_start_from=ws)
        for _ in range(3):
            model.train(input_fn=train_input_fn)
            model.evaluate(input_fn=valid_input_fn)
        # os.rename(model_dir, model_dir + str(alpha))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
