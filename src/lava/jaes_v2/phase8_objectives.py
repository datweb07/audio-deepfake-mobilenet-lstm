"""Standard objectives only. No novel mechanism is authorized by the Phase8 gate."""
import numpy as np


def group_key(label, attack, condition):
    if label not in (0, 1) or not condition:
        raise ValueError('Invalid class/condition')
    if label == 1 and attack not in tuple(f'A{i:02}' for i in range(1, 7)):
        raise ValueError('UNKNOWN attack cannot become a verified training group')
    return f'{label}:{attack if label else "REAL"}:{condition}'


def dro_update(weights, losses, rate, floor):
    """Exponentiated ascent with an explicit uniform floor; absent groups forbidden."""
    q = np.asarray(weights, dtype=np.float64)
    risks = np.asarray(losses, dtype=np.float64)
    if q.shape != risks.shape or not q.size or not np.isfinite(risks).all() or np.any(q <= 0) or not 0 <= floor < 1 or rate <= 0:
        raise ValueError('Invalid complete group risk vector')
    z = np.log(q) + rate * risks
    p = np.exp(z - z.max()); p /= p.sum()
    return (1 - floor) * p + floor / len(p)


def bce(labels, probabilities):
    import tensorflow as tf
    y = tf.cast(tf.reshape(labels, [-1]), tf.float32)
    p = tf.clip_by_value(tf.reshape(probabilities, [-1]), 1e-7, 1 - 1e-7)
    return -(y * tf.math.log(p) + (1 - y) * tf.math.log1p(-p))


def weighted_sum(labels, probabilities, coefficients):
    import tensorflow as tf
    return tf.reduce_sum(bce(labels, probabilities) * tf.cast(coefficients, tf.float32))


def clean_constraint(labels, probabilities, caps, strength):
    """Squared hinge on two classwise BCE surrogates, not a FAR guarantee."""
    import tensorflow as tf
    y = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    losses = bce(y, probabilities)
    terms = []
    for label in (0, 1):
        subset = tf.boolean_mask(losses, y == label)
        tf.debugging.assert_positive(tf.size(subset), 'Both classes needed for constraint')
        terms.append(tf.square(tf.nn.relu(tf.reduce_mean(subset) - caps[label])))
    return strength * tf.add_n(terms)
