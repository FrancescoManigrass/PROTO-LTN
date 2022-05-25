import os
from collections import defaultdict
from os.path import join

import numpy
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model
from tqdm import tqdm
from dataloader import test_unseen_classes, all_classes, test_seen_classes, attributes_class_matrix, train_classes, \
    ds_train, ds_test_zsl, ds_test_gzsl, all_data, ds_train2, ds_test_zsl2, ds_test_gzsl2
from metrics import Top1Accuracy, harmonicMean, metrics_dict, metrics_dict_history, scheduled_parameters
import logictensornetworks as ltn
from model import embeddingModel
import matplotlib.pyplot as plt

from nets import nets_factory
from resnet101 import resnet_v1_101

embeddingFunction = None
feature_extractor = None


def trainStep(train_feature, train_label, train_attribute, optimizer, config_file, **parameters):
    global embeddingFunction
    global feature_extractor
    # loss
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    with tf.GradientTape() as tape:
        # datagen.fit(train_feature)
        # train_feature = feature_extractor(train_feature, training=True)
        train_feature, train_label, train_attribute = next(iter(ds_train2))
        prototype = embeddingFunction(tf.gather(attributes_class_matrix, tf.cast(train_classes, tf.int32)),
                                      training=True)
        prototype.latent_dom = "prototype1"
        prototype.active_doms = ["prototype1"]

        axioms_satisfiability, prototype = embeddingFunction.axioms(train_feature, train_label,
                                                                    search_space=train_classes,
                                                                    prototype1=prototype, config_file=config_file)

        if config_file.loss == "1-aggregator":
            loss = 1. - embeddingFunction.satisfiabilityAggregation(axioms_satisfiability)
        elif config_file.loss == "log" and config_file.forAllAggregator == "Aggreg_pProd":
            loss = -tf.math.log(embeddingFunction.satisfiabilityAggregation(axioms_satisfiability))
        elif config_file.loss == "focal_loss":
            index_probabilities = tf.reshape(tf.where(tf.equal(
                tf.reshape(tf.tile(train_classes, [config_file.batch_size]),
                           [config_file.batch_size, len(train_classes)]),
                tf.cast(tf.reshape(train_label, [config_file.batch_size, 1]), tf.int32)))[:, -1],
                                             config_file.batch_size)
            """
            probabilities = tf.convert_to_tensor([embeddingFunction.isOfClass([train_feature, prototype]).numpy()[f][
                                 numpy.array([list(train_classes.numpy()).index(f) for f in train_label.numpy()])[f]]
                             for f in range(len(embeddingFunction.isOfClass([train_feature, prototype]).numpy()))])
            """
            index_get = tf.convert_to_tensor([[f, f] for f in range(config_file.batch_size)])
            probabilities = tf.gather_nd(
                tf.gather(embeddingFunction.isOfClass([train_feature, prototype]), index_probabilities, axis=1),
                index_get)
            loss = 1 - tf.reduce_sum(tf.multiply(
                tf.pow(1 - probabilities, 2),
                tf.math.log(probabilities)))
        elif config_file.loss == "categorical":
            cce = tf.keras.losses.CategoricalCrossentropy()
            index_probabilities = tf.reshape(tf.where(tf.equal(
                tf.reshape(tf.tile(train_classes, [config_file.batch_size]),
                           [config_file.batch_size, len(train_classes)]),
                tf.cast(tf.reshape(train_label, [config_file.batch_size, 1]), tf.int32)))[:, -1],
                                             config_file.batch_size)
            """
            probabilities = tf.convert_to_tensor([embeddingFunction.isOfClass([train_feature, prototype]).numpy()[f][
                                 numpy.array([list(train_classes.numpy()).index(f) for f in train_label.numpy()])[f]]
                             for f in range(len(embeddingFunction.isOfClass([train_feature, prototype]).numpy()))])
            """
            index_get = tf.convert_to_tensor([[f, f] for f in range(config_file.batch_size)])
            probabilities = tf.gather_nd(
                tf.gather(embeddingFunction.isOfClass([train_feature, prototype]), index_probabilities, axis=1),
                index_get)
            loss = cce(train_label, tf.convert_to_tensor(probabilities))
            print(loss)

            print("probabilities", tf.convert_to_tensor(probabilities))
        elif config_file.loss == "mse":
            mse = tf.keras.losses.MeanSquaredError()
            predictions = tf.gather(train_classes,
                                    tf.argmax(embeddingFunction.isOfClass([train_feature, prototype]), axis=-1))
            loss = mse(tf.cast(train_label, tf.float32), tf.cast(predictions, tf.float32))
        else:
            print("loss configuration error")
            return -1
        loss1 = 0.0
        loss2 = 0.0
        print(loss)
        if config_file.regularize:

            for t in embeddingFunction.trainable_variables:
                loss1 += config_file.regularization_parameter * tf.nn.l2_loss(t)

            """
            for t in feature_extractor.trainable_variables:
                loss2 += (config_file.regularization_parameter * tf.nn.l2_loss(t))
            """
        loss = (loss) + loss1  # loss1
        # embeddingFunction.trainable_variables +
    gradients = tape.gradient(loss, embeddingFunction.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, embeddingFunction.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    prototype.latent_dom = "prototype"
    prototype.active_doms = ["prototype"]
    train_feature = ltn.variable("train_feature", train_feature)
    predictions = tf.gather(train_classes, tf.argmax(embeddingFunction.isOfClass([train_feature, prototype]), axis=-1))

    match = tf.equal(predictions, tf.cast(train_label, tf.int32))
    # print("predictions",predictions)
    # print("train_label", train_label)
    # print("loss",loss)

    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


def testStep(test_feature, test_label, test_attribute, config_file, search_space, return_loss=False, **parameters):
    global embeddingFunction
    global feature_extractor
    # loss
    prototype = embeddingFunction(tf.gather(attributes_class_matrix, tf.cast(search_space, tf.int32)),
                                  training=True)

    prototype.latent_dom = "prototype1"
    prototype.active_doms = ["prototype1"]
    # test_feature = feature_extractor(test_feature)

    axioms_satisfiability, prototype = embeddingFunction.axioms(test_feature, test_label, search_space=search_space,
                                                                prototype1=prototype, config_file=config_file)
    if config_file.loss == "1-aggregator":
        loss = 1. - embeddingFunction.satisfiabilityAggregation(axioms_satisfiability)
    elif config_file.loss == "log" and config_file.forAllAggregator == "Aggreg_pProd":
        loss = -tf.math.log(embeddingFunction.satisfiabilityAggregation(axioms_satisfiability))
    elif config_file.loss == "focal_loss":
        probabilities = tf.reduce_max((embeddingFunction.isOfClass([test_feature, prototype])), axis=1)
        loss = 1 - tf.reduce_sum(tf.multiply(
            tf.pow(1 - probabilities, 1),
            tf.math.log(probabilities)))
    elif config_file.loss == "categorical":
        cce = tf.keras.losses.CategoricalCrossentropy()
        probabilities = tf.reduce_max((embeddingFunction.isOfClass([test_feature, prototype])), axis=1)
        loss = cce(test_label, probabilities)
    else:
        print("loss configuration error")
        return -1
    if config_file.regularize:
        loss1 = 0
        loss2 = 0
        if config_file.regularize:
            for t in embeddingFunction.trainable_variables:
                loss1 += config_file.regularization_parameter * tf.nn.l2_loss(t)

            """
            for t in feature_extractor.trainable_variables:
                loss2 += (config_file.regularization_parameter * tf.nn.l2_loss(t))
            """

        loss = loss + loss1  # + loss2
    if return_loss:
        metrics_dict['test_loss'](loss)

    # accuracy
    prototype.latent_dom = "prototype"
    prototype.active_doms = ["prototype"]
    test_feature = ltn.variable("test_feature", test_feature)
    predictions = tf.gather(search_space, tf.argmax(embeddingFunction.isOfClass([test_feature, prototype]), axis=-1))

    return test_label, predictions


def epochStep(epoch, optimizer, config_file):
    global embeddingFunction
    global feature_extractor
    step_train = 0
    step_test = 0
    step_gtest = 0
    for batch_elements in tqdm(ds_train2, position=0, leave=True):
        step_train += config_file.batch_size
        # print("train",step_train)
        trainStep(*batch_elements, optimizer=optimizer, config_file=config_file, **scheduled_parameters[epoch])

    true_labels_unseen = []
    predictions_zsl = []

    for batch_elements in tqdm(ds_test_zsl2, position=0, leave=True):
        step_test += config_file.batch_size
        # print("test", step_test)
        test_label, predictions = testStep(*batch_elements, search_space=test_unseen_classes, config_file=config_file,
                                           return_loss=True,
                                           **scheduled_parameters[epoch])
        true_labels_unseen.append(test_label)
        predictions_zsl.append(predictions)

    true_labels_unseen = tf.concat(true_labels_unseen, axis=0)
    predictions_zsl = tf.concat(predictions_zsl, axis=0)
    top1_zsl = Top1Accuracy(true_labels_unseen, predictions_zsl, search_space=test_unseen_classes)
    metrics_dict['top1_zsl'](top1_zsl)

    # generalized zero-shot learning (gzsl) metrics

    true_labels_all = []
    predictions_gzsl = []

    for batch_elements in tqdm(ds_test_gzsl2, position=0, leave=True):
        step_gtest += config_file.batch_size
        # print("gtest", step_gtest)
        test_label, predictions = testStep(*batch_elements, config_file=config_file, search_space=all_classes,
                                           **scheduled_parameters[epoch])
        true_labels_all.append(test_label)
        predictions_gzsl.append(predictions)

    true_labels_all = tf.concat(true_labels_all, axis=0)
    predictions_gzsl = tf.concat(predictions_gzsl, axis=0)
    top1_gzsl_unseen = Top1Accuracy(true_labels_all, predictions_gzsl, search_space=test_unseen_classes)
    metrics_dict['top1_gzsl_unseen'](top1_gzsl_unseen)
    top1_gzsl_seen = Top1Accuracy(true_labels_all, predictions_gzsl, search_space=test_seen_classes)
    metrics_dict['top1_gzsl_seen'](top1_gzsl_seen)
    H_gzsl = harmonicMean(top1_gzsl_unseen, top1_gzsl_seen)
    metrics_dict['H_gzsl'](H_gzsl)


def train(
        config_file,
        csv_path=None

):
    global embeddingFunction
    global feature_extractor
    """
    Args:
        epochs: int, number of training epochs.
        metrics_dict: dict, {"metrics_label": tf.keras.metrics instance}.
        ds_train: iterable dataset, e.g. using tf.data.Dataset.
        ds_test: iterable dataset, e.g. using tf.data.Dataset.
        train_step: callable function. the arguments passed to the function
            are the itered elements of ds_train.
        test_step: callable function. the arguments passed to the function
            are the itered elements of ds_test.
        csv_path: (optional) path to create a csv file, to save the metrics.
        scheduled_parameters: (optional) a dictionary that returns kwargs for
            the train_step and test_step functions, for each epoch.
            Call using scheduled_parameters[epoch].
    """


    train_feature, train_label, train_attribute = next(ds_train2.as_numpy_iterator())
    embeddingFunction = embeddingModel(config_file=config_file)
    prototype_initialization = embeddingFunction(attributes_class_matrix)
    prototype = embeddingFunction(tf.gather(attributes_class_matrix, tf.cast(train_classes, tf.int32)),
                                  training=True)
    prototype.latent_dom = "prototype1"
    prototype.active_doms = ["prototype1"]
    axioms_satisfiability, prototype = embeddingFunction.axioms(train_feature, train_label, search_space=train_classes,
                                                                prototype1=prototype, config_file=config_file)
    embeddingFunction.satisfiabilityAggregation(axioms_satisfiability)
    if config_file.pretrained:
        embeddingFunction.load_weights(
            config_file.weights)


    template = "Epoch {}"
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label
    if csv_path is not None:
        csv_file = open(csv_path, "w+")
        headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
        csv_file.write(headers + "\n")

    # learning_rate=scheduled_parameters[0]['learning_rate']
    best_loss = 50000000000000000000000
    best_accuracy = 0
    for epoch in range(config_file.epochs):
        print("learning_rate", config_file.learning_rate)
        print("epoch", epoch)
        optimizer = tf.keras.optimizers.Adam(config_file.learning_rate)

        for metrics in metrics_dict.values():
            metrics.reset_states()

        epochStep(epoch, optimizer, config_file=config_file)

        metrics_results = [metrics.result() for metrics in metrics_dict.values()]
        print(template.format(epoch, *metrics_results))
        if float(metrics_results[0]) < best_loss:
            best_loss = float(metrics_results[0])

        if metrics_dict['train_accuracy'].result() <= best_accuracy:
            config_file.learning_rate = config_file.learning_rate * 0.95
        else:
            best_accuracy = metrics_dict['train_accuracy'].result()

        for metrics in metrics_dict_history.keys():
            metrics_dict_history[metrics] += [metrics_dict[metrics].result().numpy()]

        if csv_path is not None:
            csv_file.write(csv_template.format(epoch, *metrics_results) + "\n")
            csv_file.flush()


    if csv_path is not None:
        csv_file.close()

    if config_file.neptune_flag:
        output_path = join("../training_results", config_file.neptune_experiemnt.experiment._experiments_stack[0].id)
    else:
        output_path = join("../training_results", "prova")
    os.makedirs(output_path, exist_ok=True)

    embeddingFunction.save_weights(join(output_path, "weights.h5"))

    prototype = embeddingFunction(attributes_class_matrix)
    prototype_df = pd.DataFrame(prototype.numpy())
    prototype_df.to_csv(join(output_path, 'prototypes.csv'))
    prototype_df.to_excel(join(output_path, 'prototypes.xlsx'))
