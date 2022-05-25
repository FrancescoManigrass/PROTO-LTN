
import os
from collections import defaultdict

import tensorflow as tf


def Top1Accuracy(true_labels, predictions, search_space):
    batch_classes, idx, counts = tf.unique_with_counts(true_labels)
    predictions = tf.cast(predictions, tf.float64)
    true_labels = tf.cast(true_labels, tf.float64)
    search_space = tf.cast(search_space, tf.float64)

    per_class_accuracies = [tf.reduce_mean(
        tf.cast(tf.equal(tf.boolean_mask(predictions, tf.equal(true_labels, label)), label), tf.float64)) for label
        in tf.cast(batch_classes,tf.float64) if label in search_space]

    return tf.math.reduce_mean(tf.stack(per_class_accuracies))


def harmonicMean(a, b):
    return 2 * a * b / (a + b)


metrics_dict = {
        'train_loss': tf.keras.metrics.Mean(name="train_loss"),
        'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
        'test_loss': tf.keras.metrics.Mean(name="test_loss"),
        'top1_zsl': tf.keras.metrics.Mean(name="top1_zsl"),
        'top1_gzsl_unseen': tf.keras.metrics.Mean(name="top1_gzsl_unseen"),
        'top1_gzsl_seen': tf.keras.metrics.Mean(name="top1_gzsl_seen"),
        'H_gzsl': tf.keras.metrics.Mean(name="H_gzsl")
    }
metrics_dict_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'top1_zsl': [],
        'top1_gzsl_unseen': [],
        'top1_gzsl_seen': [],
        'H_gzsl': []
    }

scheduled_parameters = defaultdict(lambda: {})