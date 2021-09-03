import cv2
import numpy as np
import argparse
import os
import xml.etree.ElementTree as Et
from glob import glob
from tqdm import tqdm
from tkinter.filedialog import askdirectory
from tkinter import Tk


SEQS = ['T2', 'F']
SLIDENUMS = range(7, 12)
IMGEXT = '.jpg'
LABEXT = '.xml'
COLORMAP = {'focus': (0, 255, 255), 'lesion': (255, 0, 255)}
WIDTH = 750

BRIGHTNESSCTR = 0.0
BRIGHTNESSMIN = -200.0
BRIGHTNESSMAX = 200.0
BRIGHTNESSDEL = 0.5

CONTRASTCTR = 1.0
CONTRASTMIN = 0.0
CONTRASTMAX = 5.0
CONTRASTDEL = 0.005


def adjust_img(img, size=None, brightness=0.0, contrast=1.0, labels=None):
    # resize
    xs, ys = 1.0, 1.0
    if size is not None:
        xs = size[0] / img.shape[1]
        ys = size[1] / img.shape[0]
        img = cv2.resize(img, size)
    else:
        img = img.copy()

    # brightness & contrast
    if brightness != 0.0 or contrast != 1.0:
        # img = np.clip(img * contrast + brightness, 0, 255).astype('uint8')
        img = cv2.addWeighted(img, contrast, img, 0, brightness)

    # draw labels
    if labels is not None:
        for label in labels:
            name, xmin, ymin, xmax, ymax = label
            xmin = int(xmin * xs)
            ymin = int(ymin * ys)
            xmax = int(xmax * xs)
            ymax = int(ymax * ys)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), COLORMAP[name], 2)
            cv2.putText(img, name[0].upper(), (xmin, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORMAP[name], 2)

    return img


def merge_slides(t2_imgs, labelsets, fl_imgs, plot_range, casename, brightness=0.0, contrast=1.0, draw_label=True):
    # target size
    h, w = t2_imgs[0].shape[0:2]
    width = WIDTH
    if len(plot_range) == 1:
        width = WIDTH * 2
    height = int(width * h / w)

    t2_outs = []
    fl_outs = []
    for i in plot_range:
        labels = labelsets[i] if draw_label else None
        t2 = adjust_img(t2_imgs[i], size=(width, height), brightness=brightness, contrast=contrast, labels=labels)
        cv2.putText(t2, f'{casename} #{i}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        t2_outs.append(t2)

        fl = adjust_img(fl_imgs[i], size=(width, height), brightness=brightness, contrast=contrast, labels=None)
        fl_outs.append(fl)

    if len(plot_range) == 1:
        final_img = np.hstack((t2_outs[0], fl_outs[0]))

    else:
        t2_hor = np.hstack(t2_outs)
        fl_hor = np.hstack(fl_outs)
        final_img = np.vstack((t2_hor, fl_hor))

    return final_img


def invalidate():
    global output
    output = None


def mouse_function(event, x, y, flags, param):
    global mousemode, brctmode, ix, iy, ibr, ict, br, ct
    if event == cv2.EVENT_RBUTTONDOWN:
        mousemode = True
        brctmode = 0
        ix, iy = x, y
        ibr, ict = br, ct
    elif event == cv2.EVENT_RBUTTONUP:
        mousemode = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if mousemode:
            dx = x - ix
            dy = y - iy

            if brctmode == 0:
                if abs(dx) > abs(dy):
                    brctmode_local = 1
                    if abs(dx) > abs(dy) + 20:
                        brctmode = 1
                else:
                    brctmode_local = 2
                    if abs(dy) > abs(dx) + 20:
                        brctmode = 2

            else:
                brctmode_local = brctmode

            if brctmode_local == 1:
                adjust_contrast(dx, ref=ict)
                adjust_brightness(0, ref=ibr)
                invalidate()
            elif brctmode_local == 2:
                adjust_contrast(0, ref=ict)
                adjust_brightness(-dy, ref=ibr)
                invalidate()


def adjust_contrast(point, ref=None, reset=False):
    global ct
    if reset:
        ct = CONTRASTCTR
        return

    ct0 = ct if ref is None else ref

    ct = ct0 + CONTRASTDEL * point
    ct = min(max(ct, CONTRASTMIN), CONTRASTMAX)


def adjust_brightness(point, ref=None, reset=False):
    global br
    if reset:
        br = BRIGHTNESSCTR
        return

    br0 = br if ref is None else ref

    br = br0 + BRIGHTNESSDEL * point
    br = min(max(br, BRIGHTNESSMIN), BRIGHTNESSMAX)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--width', required=False)
    args = vars(ap.parse_args())

    if args['width'] is not None:
        WIDTH = int(args['width'])

    Tk().withdraw()
    caseroot = askdirectory(initialdir='.', title='Choose Case Folder')
    casenum = os.path.basename(caseroot)

    # get file paths
    t2_paths = sorted(glob(os.path.join(caseroot, SEQS[0], '*' + IMGEXT)))
    fl_paths = sorted(glob(os.path.join(caseroot, SEQS[1], '*' + IMGEXT)))

    assert len(t2_paths) > 0, 'NO IMAGES. CHECK CASE NUMBER.'
    assert len(t2_paths) == len(fl_paths), 'Number of T2 and FLAIR do not match.'

    # load images
    t2_imgs = [cv2.imread(p) for p in tqdm(t2_paths)]
    fl_imgs = [cv2.imread(p) for p in tqdm(fl_paths)]

    print(f'{len(t2_imgs)} T2 slides and {len(fl_imgs)} FLAIR slides have been loaded.')

    # load annotations
    labelsets = []
    for t2_path in tqdm(t2_paths):
        bx_path = os.path.splitext(t2_path)[0] + LABEXT

        if os.path.isfile(bx_path):
            bxs = []
            with open(bx_path, 'r', encoding='UTF-8') as xml:
                tree = Et.parse(xml)
                root = tree.getroot()
                objects = root.findall('object')

                for obj in objects:
                    name = obj.find('name').text
                    bx = obj.find('bndbox')
                    xmin = int(bx.find('xmin').text)
                    ymin = int(bx.find('ymin').text)
                    xmax = int(bx.find('xmax').text)
                    ymax = int(bx.find('ymax').text)
                    bxs.append((name, xmin, ymin, xmax, ymax))

            labelsets.append(bxs)

        else:
            labelsets.append(None)

    # initial mode
    labelmode = True
    slidenums = SLIDENUMS
    br = BRIGHTNESSCTR
    ct = CONTRASTCTR
    mousemode = False
    brctmode = 0
    ix, iy = -1, -1
    ibr, ict = -1, -1
    output = None

    winname = f'{casenum[0:9]}'
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(winname, mouse_function)

    while True:
        # plot image
        if output is None:
            output = merge_slides(t2_imgs, labelsets, fl_imgs, slidenums, casenum,
                                  brightness=br, contrast=ct, draw_label=labelmode)

        cv2.imshow(winname, output)

        k = cv2.waitKey(20) & 0xFF
        if k == ord('l'):
            labelmode = not labelmode
            invalidate()
        elif k == ord('`'):
            slidenums = SLIDENUMS
            invalidate()
        elif k == ord('1'):
            slidenums = [7]
            invalidate()
        elif k == ord('2'):
            slidenums = [8]
            invalidate()
        elif k == ord('3'):
            slidenums = [9]
            invalidate()
        elif k == ord('4'):
            slidenums = [10]
            invalidate()
        elif k == ord('5'):
            slidenums = [11]
            invalidate()
        elif k == ord('z'):
            adjust_brightness(-10)
            invalidate()
        elif k == ord('c'):
            adjust_brightness(10)
            invalidate()
        elif k == ord('x'):
            adjust_brightness(0, reset=True)
            invalidate()
        elif k == ord('a'):
            adjust_contrast(-10)
            invalidate()
        elif k == ord('d'):
            adjust_contrast(10)
            invalidate()
        elif k == ord('s'):
            adjust_contrast(0, reset=True)
            invalidate()
        elif k == 27:
            break

    cv2.destroyAllWindows()

