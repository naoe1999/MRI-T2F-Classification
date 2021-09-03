import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


IMG_SIZE = (288, 384)


def main():
    # flo_dir = './data/train_960/FLO'
    # flx_dir = './data/train_960/FLX'
    #
    # dirs = [flo_dir, flx_dir]
    # labels = ['FLO', 'FLX']

    cls_1_dir = './train_src/1'
    cls_2_dir = './train_src/2'
    cls_3_dir = './train_src/3'

    dirs = [cls_1_dir, cls_2_dir, cls_3_dir]
    labels = ['1', '2', '3']

    img_list = []
    lbl_list = []

    for d, l in zip(dirs, labels):
        imgfiles = os.listdir(d)

        for imgf in imgfiles:
            if os.path.splitext(imgf)[-1].lower() not in ['.jpg', '.png']:
                continue

            img = cv2.imread(os.path.join(d, imgf))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)

            img_list.append(img)
            lbl_list.append(l)

    img_array = np.array(img_list)
    print(img_array.shape)

    img_array = img_array.reshape((img_array.shape[0], -1))
    print(img_array.shape)

    tsne = TSNE(n_components=2, random_state=0)
    res = tsne.fit_transform(img_array)

    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=lbl_list)
    plt.show()


if __name__ == '__main__':
    main()

