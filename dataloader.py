import numpy as np
import scipy.io
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image as image_utils
from config import config
import logictensornetworks as ltn
import tensorflow as tf
config_file = config()
images_train=None
images_test=None
images_test_unseen=None
labels_train =None
attributes_train = None
labels_test =None
attributes_test =None
labels_test_unseen = None
attributes_test_unseen = None

features_train= None
features_test_unseen=None
features_test=None



import cv2

def getDataset(
        base_path,
        buffer_size,
        batch_size):
    global images_train, images_test, images_test_unseen, labels_train, attributes_train, labels_test, attributes_test, labels_test_unseen, attributes_test_unseen, features_train,features_test_unseen,features_test

    res101 = scipy.io.loadmat(base_path + 'res101.mat')
    all_features = np.transpose(res101['features'])
    all_images = np.transpose(res101['image_files'])
    all_labels = res101['labels'] - 1
    all_labels=all_labels.astype(int)
    att_splits = scipy.io.loadmat(base_path + "att_splits.mat") #FOR AWA2
    classes_names = att_splits['allclasses_names']
    classes_names = [classes_names[i][0][0] for i in range(classes_names.size)]
    class_attributes_matrix = att_splits['att']
    attributes_class_matrix = np.transpose(class_attributes_matrix)
    test_unseen = att_splits['test_unseen_loc'] - 1
    test_seen = att_splits['test_seen_loc'] - 1
    test = np.concatenate((test_unseen, test_seen))
    train = att_splits['trainval_loc'] - 1
    attribute = att_splits['original_att'].T  # att for AWA2
    #attribute = att_splits['att'].T  # att FOR GBU

    features_train = all_features[train].reshape((all_features[train].shape[0], 2048))
    images_train= all_images[0][train]
    labels_train = all_labels[train].reshape(all_labels[train].shape[0])
    attributes_train = attributes_class_matrix[labels_train]
    features_test_unseen = all_features[test_unseen].reshape((all_features[test_unseen].shape[0], 2048))
    images_test_unseen = all_images[0][test_unseen]
    labels_test_unseen = all_labels[test_unseen].reshape(all_labels[test_unseen].shape[0])
    attributes_test_unseen = attributes_class_matrix[labels_test_unseen]
    features_test_seen = all_features[test_seen].reshape((all_features[test_seen].shape[0], 2048))
    labels_test_seen = all_labels[test_seen].reshape(all_labels[test_seen].shape[0])
    attributes_test_seen = attributes_class_matrix[labels_test_seen]
    features_test = all_features[test].reshape((all_features[test].shape[0], 2048))
    images_test = all_images[0][test]
    labels_test = all_labels[test].reshape(all_labels[test].shape[0])
    attributes_test = attributes_class_matrix[labels_test]
    test_unseen_classes =tf.convert_to_tensor( np.unique(labels_test_unseen))
    train_classes = tf.convert_to_tensor(np.unique(labels_train))

    images_train=images_train
    images_test = images_test
    images_test_unseen = images_test_unseen
    labels_train = tf.convert_to_tensor(labels_train, np.float32)
    attributes_train =  tf.convert_to_tensor(attributes_train, np.float32)
    labels_test =tf.convert_to_tensor(labels_test, np.float32)
    attributes_test =  tf.convert_to_tensor(attributes_test, np.float32)
    labels_test_unseen =   tf.convert_to_tensor(labels_test_unseen,  np.float32)
    attributes_test_unseen = tf.convert_to_tensor(attributes_test_unseen, np.float32)

    ds_train2 = tf.data.Dataset.from_tensor_slices((features_train, labels_train, attributes_train)).shuffle(
        buffer_size).batch(batch_size)
    ds_test_gzsl2 = tf.data.Dataset.from_tensor_slices((features_test, labels_test, attributes_test)).shuffle(
        buffer_size).batch(batch_size)
    ds_test_zsl2 = tf.data.Dataset.from_tensor_slices(
        (features_test_unseen, labels_test_unseen, attributes_test_unseen)).shuffle(buffer_size).batch(batch_size)


    all_data = {
        'attributes_class_matrix': attribute,
        'classes_names': classes_names,
        'features_train': features_train,
        'labels_train': labels_train,
        'attributes_train': attributes_train,
        'features_test': features_test,
        'labels_test': labels_test,
        'attributes_test': attributes_test,
        'features_test_seen': features_test_seen,
        'labels_test_seen': labels_test_seen,
        'attributes_test_seen': attributes_test_seen,
        'features_test_unseen': features_test_unseen,
        'labels_test_unseen': labels_test_unseen,
        'attributes_test_unseen': attributes_test_unseen,
        'test_unseen_classes': test_unseen_classes,
        'train_classes': train_classes
    }

    return ds_train, ds_test_zsl, ds_test_gzsl, all_data,ds_train2,ds_test_zsl2,ds_test_gzsl2



ds_train, ds_test_zsl, ds_test_gzsl, all_data,ds_train2,ds_test_zsl2,ds_test_gzsl2 = getDataset(
    base_path=config_file.base_path,
    buffer_size=config_file.buffer_size,
    batch_size=config_file.batch_size)

attributes_class_matrix = all_data['attributes_class_matrix']
test_unseen_classes = all_data['test_unseen_classes']
train_classes = all_data['train_classes']
test_seen_classes = train_classes
all_classes = np.concatenate((train_classes, test_unseen_classes))

attributes_class_matrix = ltn.variable("attributes_class_matrix", attributes_class_matrix)
attributes_class_matrix_train = ltn.variable("attributes_class_matrix_train",
                                             attributes_class_matrix.numpy()[train_classes])

attributes_class_matrix_test_unseen = ltn.variable("attributes_class_matrix_test_unseen",
                                                   attributes_class_matrix.numpy()[test_unseen_classes.numpy()])
not_zeros = lambda x: (1 - config_file.eps) * x + config_file.eps
not_ones = lambda x: (1 - config_file.eps) * x
