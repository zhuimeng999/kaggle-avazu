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
            embedding_table = tf.get_variable(k + '_embed_lr', [params['feature_dim'][k], params['hidden_fields']], 
                                              initializer=tf.truncated_normal_initializer(tf.sqrt(2/66)))
            embedded_var = tf.nn.embedding_lookup(embedding_table, v)
            lr_list.append(embedded_var)
        
    lr_matrix = tf.concat(lr_list, axis=1)

    with tf.variable_scope('dnn_layer'):
        net = params['activation'](lr_matrix)
        if (params.get('keep_prob', 1.0)) < 1.0 and (mode == tf.estimator.ModeKeys.TRAIN):
            net = tf.nn.dropout(net, params['keep_prob'])
        for _ in range(params['num_layer'] - 1):
            net = tf.layers.dense(net, units=params['hidden_fields'], activation=params['activation'],
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2/66)),
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)
            if (params.get('keep_prob', 1.0)) < 1.0 and (mode == tf.estimator.ModeKeys.TRAIN):
                net = tf.nn.dropout(net, params['keep_prob'])

    logits = tf.reduce_sum(net, axis=1)
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
