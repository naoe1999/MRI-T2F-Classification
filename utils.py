from tqdm import tqdm
import xml.etree.ElementTree as Et
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import sparse_categorical_crossentropy, mse


# tensorflow model functions

def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def multi_loss(y_true, y_pred):
    # assert y_true.shape[-1] == 5
    # assert y_pred.shape[-1] == 6

    class_label = y_true[..., 0]
    class_pred = y_pred[..., 0:2]

    roi_true = y_true[..., 1:]
    roi_pred = y_pred[..., 2:]

    # classification loss
    class_loss = sparse_categorical_crossentropy(class_label, class_pred)

    # regression loss
    reg_loss_mask = K.mean(K.cast(K.not_equal(roi_true, -1.0), 'float32'), axis=-1)
    reg_loss = mse(roi_true, roi_pred) * reg_loss_mask

    loss = class_loss + reg_loss * 2.0
    loss = K.mean(loss)

    # for test
    # K.print_tensor(y_true, message='y true = ')
    # K.print_tensor(y_pred, message='y pred = ')
    # K.print_tensor(class_loss, message='class loss = ')
    # K.print_tensor(reg_loss, message='reg_loss = ')
    # K.print_tensor(loss, message='loss = ')

    return loss


def accuracy(y_true, y_pred):
    # assert y_true.shape[-1] == 5
    # assert y_pred.shape[-1] == 6

    # classification accuracy only
    class_label = K.cast(y_true[..., 0], 'int64')
    class_pred = y_pred[..., 0:2]

    correct = K.cast(K.equal(class_label, K.argmax(class_pred)), 'float32')
    acc = K.mean(correct)
    return acc


#
# data loader functions

def load_images(file_paths):
    imgs = [image.img_to_array(image.load_img(f, target_size=(384, 288))) for f in tqdm(file_paths)]
    return np.asarray(imgs)


def load_annotations(file_paths, label=None, keep_label=True, default=-1.0):
    annos = []
    for f in tqdm(file_paths):
        with open(f, 'r', encoding='UTF-8') as xml:
            tree = Et.parse(xml)
            root = tree.getroot()
            objects = root.findall('object')
            size = root.find('size')
            W = float(size.find('width').text)
            H = float(size.find('height').text)

            bxs = []
            for obj in objects:
                name = obj.find('name').text
                bx = obj.find('bndbox')
                xmin = float(bx.find('xmin').text) / W
                ymin = float(bx.find('ymin').text) / H
                xmax = float(bx.find('xmax').text) / W
                ymax = float(bx.find('ymax').text) / H

                xctr = (xmin + xmax) / 2.0
                yctr = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin

                if label is not None and name != label:
                    continue

                bxs.append((xctr, yctr, width, height, name)[:4 + int(keep_label)])

            if len(bxs) == 0:
                bxs.append((default, default, default, default, 'NONE')[:4 + int(keep_label)])
        annos.append(bxs)
    return annos


def pick_only_one_annotation_each(annos):
    new_annos = []

    # area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    area = lambda b: b[2] * b[3]

    for bxs in annos:
        areas = np.asarray([area(bx) for bx in bxs])
        new_annos.append(bxs[np.argmax(areas)])

    return np.asarray(new_annos)


def merge_data_and_get_labels(positive_imgs, negative_imgs, positive_annos, negative_anno_fill_value=-1.0):
    n_pos = len(positive_imgs)
    n_neg = len(negative_imgs)
    assert len(positive_annos) == n_pos

    merged_imgs = np.concatenate((positive_imgs, negative_imgs), axis=0)

    neg_anno_shape = (n_neg,) + positive_annos.shape[1:]
    negative_annos = np.ones(neg_anno_shape) * negative_anno_fill_value
    merged_annos = np.concatenate((positive_annos, negative_annos), axis=0)

    labels = np.concatenate((np.zeros((n_pos, 1)), np.ones((n_neg, 1))), axis=0)    # positive index: 0
    merged_labels = np.concatenate((labels, merged_annos), axis=1)

    return merged_imgs, merged_labels

