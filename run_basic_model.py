import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from glob import glob


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

    # 1. data
    # train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, brightness_range=[0.8, 1.2])
    train_datagen = ImageDataGenerator(rescale=1./255, brightness_range=[0.8, 1.2])
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(384, 288), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR, target_size=(384, 288), batch_size=BATCH_SIZE, class_mode='categorical'
    )


    # 2. model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=[384, 288, 3])

    x = vgg16.get_layer('block5_conv3').output
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dropout(DROPOUT_RATE, name='dropout')(x)
    outputs = keras.layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=vgg16.inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['acc'])


    # 3. training
    if TRAIN_MODE:
        # run training
        history = model.fit(
            x=train_generator, epochs=EPOCH, validation_data=valid_generator, validation_steps=10
        )

        # save model
        execution_name = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
        model_path = os.path.join(MODEL_ROOT, f'weights_{execution_name}.h5')
        model.save(model_path)
        model.summary()

        # draw training histor0y curve
        acc = history.history['acc']
        val_acc = history.history['val_acc']
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
        model = keras.models.load_model(model_path)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['acc'])


    # 4. evaluation
    val_loss, val_acc = model.evaluate(x=valid_generator)
    print('validation loss:', val_loss)
    print('validation accuracy:', val_acc)


    # 5. class activation map
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
                pred = pred[0]

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

