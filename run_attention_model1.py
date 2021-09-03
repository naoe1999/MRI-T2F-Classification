from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from utils import init_gpu, load_images, load_annotations, \
    pick_only_one_annotation_each, merge_data_and_get_labels, \
    multi_loss, accuracy


TRAIN_MODE = True
SAVE_CAM = True
POSITIVE_IDX = 0
LEARNING_RATE = 1e-5
EPOCH = 30
BATCH_SIZE = 8
DROPOUT_RATE = 0.5
TRAIN_DIR = './data/train_960'
VALID_DIR = './data/test'
MODEL_ROOT = './result/weights'
PLOT_ROOT = './result/plots'
CAM_ROOT = './result/CAMs'
EXEC_NAME_TO_RESTORE = ''



if __name__ == '__main__':
    init_gpu()

    #############################################################################################
    #
    # 1. list image & annotation files
    #
    #############################################################################################
    train_flo_img_files = sorted(glob(os.path.join(TRAIN_DIR, 'FLO', '*.jpg')))
    train_flo_anno_files = sorted(glob(os.path.join(TRAIN_DIR, 'FLO', '*.xml')))

    train_flx_img_files = sorted(glob(os.path.join(TRAIN_DIR, 'FLX', '*.jpg')))

    valid_flo_img_files = sorted(glob(os.path.join(VALID_DIR, 'FLO', '*.jpg')))
    valid_flo_anno_files = sorted(glob(os.path.join(VALID_DIR, 'FLO', '*.xml')))

    valid_flx_img_files = sorted(glob(os.path.join(VALID_DIR, 'FLX', '*.jpg')))

    # assertion check
    assert len(train_flo_img_files) == len(train_flo_anno_files), 'Train FLO image & annotation files mismatch.'
    for i in range(len(train_flo_img_files)):
        assert os.path.splitext(os.path.basename(train_flo_img_files[i]))[0] == \
               os.path.splitext(os.path.basename(train_flo_anno_files[i]))[0], \
               'Train FLO image & annotation files mismatch.'

    assert len(valid_flo_img_files) == len(valid_flo_anno_files), 'Test FLO image & annotation files mismatch.'
    for i in range(len(valid_flo_img_files)):
        assert os.path.splitext(os.path.basename(valid_flo_img_files[i]))[0] == \
               os.path.splitext(os.path.basename(valid_flo_anno_files[i]))[0], \
               'Test FLO image & annotation files mismatch.'

    #############################################################################################
    #
    # 2. load image
    #
    #############################################################################################
    train_flo_imgs = load_images(train_flo_img_files)
    train_flx_imgs = load_images(train_flx_img_files)

    valid_flo_imgs = load_images(valid_flo_img_files)
    valid_flx_imgs = load_images(valid_flx_img_files)

    print('train | FLO | images :', train_flo_imgs.shape)
    print('train | FLX | images :', train_flx_imgs.shape)
    print('test | FLO | images :', valid_flo_imgs.shape)
    print('test | FLO | images :', valid_flx_imgs.shape)
    print('loading images completed.')

    #############################################################################################
    #
    # 3. load annotation
    #
    #############################################################################################
    train_flo_annos = load_annotations(train_flo_anno_files, label='lesion', keep_label=False)
    train_flo_annos = pick_only_one_annotation_each(train_flo_annos)

    valid_flo_annos = load_annotations(valid_flo_anno_files, label='lesion', keep_label=False)
    valid_flo_annos = pick_only_one_annotation_each(valid_flo_annos)

    print('train | FLO | annotations :', train_flo_annos.shape)
    print('test | FLO | annotations :', valid_flo_annos.shape)
    print('loading annotations completed.')

    #############################################################################################
    #
    # 4. build data flow
    #
    #############################################################################################
    train_imgs, train_labels = merge_data_and_get_labels(train_flo_imgs, train_flx_imgs, train_flo_annos, -1)
    valid_imgs, valid_labels = merge_data_and_get_labels(valid_flo_imgs, valid_flx_imgs, valid_flo_annos, -1)

    print('train image, label:', train_imgs.shape, train_labels.shape)
    print('test image, label:', valid_imgs.shape, valid_labels.shape)

    train_datagen = ImageDataGenerator(rescale=1./255, brightness_range=[0.8, 1.2])
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_imgs, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    valid_generator = valid_datagen.flow(valid_imgs, valid_labels, batch_size=BATCH_SIZE, shuffle=True)

    #############################################################################################
    #
    # 5. build model
    #
    #############################################################################################
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(384, 288, 3))

    input = vgg16.input
    featuremap = vgg16.get_layer('block5_conv3').output
    featuremap_pooled = vgg16.output
    # featuremap_pooled = vgg16.get_layer('block4_conv3').output

    # classification header
    x = GlobalAveragePooling2D(name='gap')(featuremap)
    x = Dropout(DROPOUT_RATE)(x)
    cls_output = Dense(2, activation='softmax', name='class_prediction')(x)

    # regression header
    x = Flatten(name='flatten')(featuremap_pooled)
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    roi_output = Dense(4, activation='sigmoid', name='roi_prediction')(x)

    output = concatenate([cls_output, roi_output], axis=-1)

    model = Model(inputs=input, outputs=output)
    model.summary()

    model.compile(loss=multi_loss,
                  optimizer=Adam(lr=LEARNING_RATE),
                  metrics=[accuracy])

    #############################################################################################
    #
    # 6. train
    #
    #############################################################################################
    if TRAIN_MODE:
        history = model.fit(
            x=train_generator, epochs=EPOCH,
            validation_data=valid_generator, validation_steps=len(valid_imgs)//BATCH_SIZE
        )

        # save model
        execution_name = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
        model_path = os.path.join(MODEL_ROOT, f'weights_{execution_name}.h5')
        model.save(model_path)
        model.summary()

        # draw training histor0y curve
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plot_dir = os.path.join(PLOT_ROOT, execution_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.figure()
        plt.plot(epochs, acc, 'b-', label='Training acc')
        plt.plot(epochs, val_acc, 'r-', label='Validation acc')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'acc.png'))
        plt.show()

        plt.figure()
        plt.plot(epochs, loss, 'b-', label='Training loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'loss.png'))
        plt.show()

    else:
        # restore model
        execution_name = EXEC_NAME_TO_RESTORE
        model_path = os.path.join(MODEL_ROOT, f'weights_{execution_name}.h5')
        model = load_model(model_path, custom_objects={'multi_loss': multi_loss, 'accuracy': accuracy})
        model.summary()

        model.compile(loss=multi_loss,
                      optimizer=Adam(lr=LEARNING_RATE),
                      metrics=[accuracy])

    #############################################################################################
    #
    # 7. evaluate
    #
    #############################################################################################
    val_loss, val_acc = model.evaluate(x=valid_generator)
    print('validation loss:', val_loss)
    print('validation accuracy:', val_acc)

    #############################################################################################
    #
    # 8. class activation map
    #
    #############################################################################################
    if SAVE_CAM:

        # set input image dirs
        img_dir_gts = [
            (os.path.join(TRAIN_DIR, 'FLO'), 0),
            (os.path.join(TRAIN_DIR, 'FLX'), 1),
            (os.path.join(VALID_DIR, 'FLO'), 0),
            (os.path.join(VALID_DIR, 'FLX'), 1)
        ]

        # set output CAM dirs
        cam_dir_root = os.path.join(CAM_ROOT, execution_name)
        cam_dir_pairs = [
            [os.path.join(cam_dir_root, 'train', 'FLO', sf) for sf in ['S', 'F']],
            [os.path.join(cam_dir_root, 'train', 'FLX', sf) for sf in ['S', 'F']],
            [os.path.join(cam_dir_root, 'test', 'FLO', sf) for sf in ['S', 'F']],
            [os.path.join(cam_dir_root, 'test', 'FLX', sf) for sf in ['S', 'F']]
        ]
        for sdir, fdir in cam_dir_pairs:
            os.makedirs(sdir, exist_ok=True)
            os.makedirs(fdir, exist_ok=True)

        # model for CAM prediction
        gradModel = Model(inputs=model.inputs,
                          outputs=[model.get_layer('block5_conv3').output, model.output])

        # get Grad-CAM for all images
        for img_dir_gt, cam_dir_pair in zip(img_dir_gts, cam_dir_pairs):
            img_dir, gt = img_dir_gt
            img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))

            for img_path in tqdm(img_paths):
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                img = image.load_img(img_path, target_size=(384, 288))
                x = image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)

                with tf.GradientTape() as tape:
                    (conv, pred) = gradModel(x)
                    y = pred[:, POSITIVE_IDX]

                grads = tape.gradient(y, conv)[0]
                conv = conv[0]
                pred = pred[0, 0:2]

                weights = tf.reduce_mean(grads, axis=(0, 1))
                cam = tf.reduce_sum(tf.multiply(weights, conv), axis=-1)
                cam = tf.maximum(cam, 0)

                heatmap = cv2.resize(cam.numpy(), (288, 384))

                numer = heatmap - heatmap.min()
                denom = (heatmap.max() - heatmap.min())
                heatmap = numer / denom
                heatmap = (heatmap * 255).astype("uint8")
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                img = image.img_to_array(img).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

                # check if prediction is correct
                pred = pred.numpy()
                correct = np.argmax(pred) == gt
                cam_dir = cam_dir_pair[0 if correct else 1]
                img_name = f'{img_name}_{pred[POSITIVE_IDX]:.2f}.jpg'

                cv2.imwrite(os.path.join(cam_dir, img_name), output)

