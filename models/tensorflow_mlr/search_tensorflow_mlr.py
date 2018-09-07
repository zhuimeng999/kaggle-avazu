import tensorflow as tf
import os
from models.tensorflow_dnn.tensorflow_dnn import MyModel
from models.dataset_fn.ctr_dataset import get_feature_dim, get_input_fn

tf.app.flags.DEFINE_integer("thresh_hold", 0, 'thresh_hold')
FLAGS = tf.app.flags.FLAGS


def main(_):
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(curr_dir, 'logdir_s')
    train_path = os.path.join(curr_dir, '../../preprocess/output/train_shuffle_' + str(FLAGS.thresh_hold) + '.csv')
    valid_path = os.path.join(curr_dir, '../../preprocess/output/valid_raw_' + str(FLAGS.thresh_hold) + '.csv')

    batch_size = 2048
    train_input_fn = get_input_fn(train_path, batch_size, 1, 10000)
    valid_input_fn = get_input_fn(valid_path, batch_size, 1, 10000)

    # alpha_list = [0.00001, 0.0001, 0.001, 0.01]
    hidden_fields_list = [3, 6, 9]
    keep_prob_list = [0.8, 0.85, 0.9, 0.95]
    params = {
        'alpha': 0,
        'feature_dim': get_feature_dim(FLAGS.thresh_hold),
        'hidden_fields': 5,
        'keep_prob': 0.9,
        'num_layer': 2,
        'activation': tf.nn.tanh
    }
    # ws = tf.estimator.WarmStartSettings(
    #     ckpt_to_initialize_from=warm_start_dir,
    #     vars_to_warm_start=".*input_layer.*")
    ws = None
    for keep_prob in keep_prob_list:
        for hidden_fields in hidden_fields_list:
            params['keep_prob'] = keep_prob
            params['hidden_fields'] = hidden_fields
            model_dir = model_dir + str(keep_prob) + '_' + str(hidden_fields)
            model = MyModel(model_dir=model_dir, params=params, warm_start_from=ws)
            for _ in range(3):
                model.train(input_fn=train_input_fn)
                model.evaluate(input_fn=valid_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
