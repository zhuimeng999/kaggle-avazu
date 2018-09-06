import tensorflow as tf

def my_model(features, labels, mode, params):
    alpha = params['alpha']
    if alpha > 0:
        regularizer = tf.contrib.layers.l1_regularizer(alpha)
    else:
        regularizer = None
    
    lr_list = []
    with tf.variable_scope('input_layer'):
        for k, v in features.items():
            embedding_table = tf.get_variable(k + '_embed_lr', [params['feature_dim'][k], 1], 
                                              initializer=tf.truncated_normal_initializer(tf.sqrt(2/66)), regularizer=regularizer)
            embedded_var = tf.nn.embedding_lookup(embedding_table, v)
            lr_list.append(embedded_var)
            
        lr_bias = tf.get_variable('lr_bias', shape=(), dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(tf.sqrt(2/66)), regularizer=regularizer)
    lr_matrix = tf.concat(lr_list, axis = 1)
    logits = tf.reduce_sum(lr_matrix, axis=1) + lr_bias
    # Compute predictions.
    predicted_classes = tf.cast(tf.greater(logits, 0), tf.int64)
    probabilities = tf.nn.sigmoid(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': probabilities,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.l
    loss = tf.losses.log_loss(labels=labels, predictions=probabilities)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    loss = loss + tf.losses.get_regularization_loss()
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    learn_rate = params.get('learn_rate', 0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    adam_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())


    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    sgd_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if ('optimizer' in params) and (params['optimizer'] == 'sgd'):
        train_op = sgd_op
    else:
        train_op = adam_op

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class MyModel(tf.estimator.Estimator):
    def __init__(self, model_dir=None, config=None, params=None,
                 warm_start_from=None):
        super().__init__(my_model, model_dir=model_dir, config=config, params=params,
                         warm_start_from=warm_start_from)
        