from .__omni__ import *


# : BinaryCrossEntropy
# ----------------------------------------------------------------------------------------------------------
def BinaryCrossEntropy(outputs: array | Tensor, targets: array | Tensor):
    outputs = tf.clip_by_value(outputs, 1E-15, 1.0)
    loss = tf.reduce_mean(tf.losses.binary_crossentropy(targets, outputs))
    return loss if type(targets) is Tensor else loss.numpy()


def _v2_BinaryCrossEntropy(outputs: Tensor, targets: Tensor, *, epsilon: float = 1E-15):
    outputs = tf.clip_by_value(outputs, epsilon, 1.0)
    loss = tf.where(tf.equal(targets, 1.0), -tf.math.log(outputs), -tf.math.log(1.0 - outputs))
    return tf.reduce_mean(loss)


# : BinaryAccuracy
# ----------------------------------------------------------------------------------------------------------
def BinaryAccuracy(outputs: array | Tensor, targets: array | Tensor, threshold: float = 0.5):
    P, N = outputs[targets == 1], outputs[targets == 0]
    TP, TN = len(P[tf.greater_equal(P, threshold)]), len(N[tf.less(N, threshold)])
    accuracy = tf.constant((TP + TN) / len(targets))
    return accuracy if type(targets) is Tensor else accuracy.numpy()


# : CrossEntropy
# ----------------------------------------------------------------------------------------------------------
def CrossEntropy(outputs: array | Tensor, targets: array | Tensor):
    outputs = tf.clip_by_value(outputs, 1E-15, 1.0)
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(targets, outputs))
    return loss if type(targets) is Tensor else loss.numpy()


def _v2_CrossEntropy(outputs: Tensor, targets: Tensor, *, epsilon: float = 1E-15):
    outputs = tf.clip_by_value(outputs, epsilon, 1.0)
    loss = tf.reduce_sum(-targets * tf.math.log(outputs), axis=-1, keepdims=True)
    return tf.reduce_mean(loss)


# : CategoricalAccuracy
# ----------------------------------------------------------------------------------------------------------
def CategoricalAccuracy(outputs: array | Tensor, targets: array | Tensor):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(
        tf.argmax(targets, axis=-1), tf.argmax(outputs, axis=-1)), tf.int32)) / len(targets)
    return accuracy if type(targets) is Tensor else accuracy.numpy()


# : BinaryFocalLoss
# ----------------------------------------------------------------------------------------------------------
def BinaryFocalLoss(outputs: array | Tensor, targets: array | Tensor, *,
                    alpha: float = 0.25, gamma: float = 2.0, epsilon: float = 1E-15):
    outputs = tf.clip_by_value(outputs, epsilon, 1.0)
    binary_cross_entropy = tf.where(
        tf.equal(targets, 1.0), -tf.math.log(outputs), -tf.math.log(1.0 - outputs))
    factor = tf.where(
        tf.equal(targets, 1.0),
        alpha * tf.pow(1 - outputs, gamma), (1 - alpha) * tf.pow(outputs, gamma))
    loss = tf.reduce_mean(factor * binary_cross_entropy)
    return loss if type(targets) is Tensor else loss.numpy()


# : CenterLoss
# ----------------------------------------------------------------------------------------------------------
def CenterLoss(outputs: array | Tensor, targets: array | Tensor,
              *, alpha: float = 1.0, gamma: float = 2.0, epsilon: float = 1E-15):
    outputs = tf.clip_by_value(outputs, epsilon, 1. - epsilon)
    targets = tf.clip_by_value(targets, epsilon, 1. - epsilon)
    cross_entropy = -targets * tf.math.log(outputs)
    loss = alpha * tf.pow(1 - outputs, gamma) * cross_entropy
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss if type(targets) is Tensor else loss.numpy()

# : FocalLoss
# ----------------------------------------------------------------------------------------------------------
def FocalLoss(outputs: array | Tensor, targets: array | Tensor,
              *, alpha: float = 1.0, gamma: float = 2.0, epsilon: float = 1E-15):
    outputs = tf.clip_by_value(outputs, epsilon, 1. - epsilon)
    targets = tf.clip_by_value(targets, epsilon, 1. - epsilon)
    cross_entropy = -targets * tf.math.log(outputs)
    loss = alpha * tf.pow(1 - outputs, gamma) * cross_entropy
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss if type(targets) is Tensor else loss.numpy()


# : Divergence (JS)
# ----------------------------------------------------------------------------------------------------------
def Divergence(a: array | Tensor, b: array | Tensor):
    a = tf.clip_by_value(a, 1E-15, 1.0)
    b = tf.clip_by_value(b, 1E-15, 1.0)
    divergence = tf.reduce_mean(
        1 / 2 * tf.losses.kullback_leibler_divergence(a, (a + b) / 2) +
        1 / 2 * tf.losses.kullback_leibler_divergence(b, (a + b) / 2))
    return divergence if type(a) is Tensor else divergence.numpy()


def KL_Divergence(a: array | Tensor, b: array | Tensor):
    a = tf.clip_by_value(a, 1E-15, 1.0)
    b = tf.clip_by_value(b, 1E-15, 1.0)
    divergence = tf.reduce_mean(tf.losses.kullback_leibler_divergence(a, b))
    return divergence if type(a) is Tensor else divergence.numpy()


# : group_predictions
# ----------------------------------------------------------------------------------------------------------
def group_predictions(outputs: array | Tensor, targets: array | Tensor, threshold: float = 0.5):
    P = outputs[targets == 1]  # Positive Samples' Predictions ...
    N = outputs[targets == 0]  # Negative Samples' Predictions ...
    TP = len(P[tf.greater_equal(P, threshold)])
    FN = len(P[tf.less(P, threshold)])
    FP = len(N[tf.greater_equal(N, threshold)])
    TN = len(N[tf.less(N, threshold)])
    return TP, FN, FP, TN


# : F_score
# ----------------------------------------------------------------------------------------------------------
def F_score(outputs: array | Tensor, targets: array | Tensor, threshold: float = 0.5):
    TP, FN, FP, _ = group_predictions(outputs, targets, threshold)
    return 2 * TP / (2 * TP + FP + FN)
