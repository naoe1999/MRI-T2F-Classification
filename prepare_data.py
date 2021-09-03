import os
import csv
from shutil import copyfile
from tqdm import tqdm


SOURCEFILEFORMAT = 'I{0:07d}'
TARGETFILEFORMAT = '{0:03d}'
IMAGEEXT = '.jpg'
LABELEXT = '.xml'
ROI = range(7, 12)


def copy_with_metadata(src_dir, tgt_dir, cls_str, meta_csv):
    srcfiles = []
    tgtfiles = []

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    with open(meta_csv, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            caseno = line[0]
            imgclss = line[1:]

            for idx in ROI:
                if imgclss[idx].strip() == cls_str:
                    srcbase = os.path.join(src_dir, caseno, 'T2', SOURCEFILEFORMAT.format(idx))
                    srcimg = srcbase + IMAGEEXT
                    srclbl = srcbase + LABELEXT

                    tgtbase = os.path.join(tgt_dir, caseno + TARGETFILEFORMAT.format(idx))
                    tgtimg = tgtbase + IMAGEEXT
                    tgtlbl = tgtbase + LABELEXT

                    srcfiles.append(srcimg)
                    tgtfiles.append(tgtimg)

                    srcfiles.append(srclbl)
                    tgtfiles.append(tgtlbl)

    for src, tgt in tqdm(zip(srcfiles, tgtfiles)):
        copyfile(src, tgt)


def copy_all_cases(src_dir, tgt_dir):
    srcfiles = []
    tgtfiles = []

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    casenos = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for caseno in casenos:
        for idx in ROI:
            srcbase = os.path.join(src_dir, caseno, 'T2', SOURCEFILEFORMAT.format(idx))
            srcimg = srcbase + IMAGEEXT

            tgtbase = os.path.join(tgt_dir, caseno + TARGETFILEFORMAT.format(idx))
            tgtimg = tgtbase + IMAGEEXT

            srcfiles.append(srcimg)
            tgtfiles.append(tgtimg)

    for src, tgt in tqdm(zip(srcfiles, tgtfiles)):
        copyfile(src, tgt)


def rearrange_data(case_root_dir, tgt_root_dir, meta_csv):
    copylist = []

    if not os.path.exists(tgt_root_dir):
        os.makedirs(tgt_root_dir)

    with open(meta_csv, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            caseno = line[0]
            imgclss = line[1:]

            for idx in ROI:
                cls = imgclss[idx].strip()
                srcbase = os.path.join(case_root_dir, caseno, 'T2', SOURCEFILEFORMAT.format(idx))

                tgt_dir = os.path.join(tgt_root_dir, cls)
                if not os.path.exists(tgt_dir):
                    os.mkdir(tgt_dir)

                tgtbase = os.path.join(tgt_dir, caseno + TARGETFILEFORMAT.format(idx))

                srcimg = srcbase + IMAGEEXT
                tgtimg = tgtbase + IMAGEEXT
                copylist.append((srcimg, tgtimg))

                if cls in ['1', '2']:
                    srclbl = srcbase + LABELEXT
                    tgtlbl = tgtbase + LABELEXT
                    copylist.append((srclbl, tgtlbl))

    for src, tgt in tqdm(copylist):
        copyfile(src, tgt)



if __name__ == '__main__':

    # class 1
    print('copying class 1 files ...')
    class1_srcdir = os.path.join(os.getcwd(), 'train_src', 'cases', 'class1')
    class1_tgtdir = os.path.join(os.getcwd(), 'train_src', '1')
    class1_csv = class1_srcdir + '.csv'
    copy_with_metadata(class1_srcdir, class1_tgtdir, '1', class1_csv)
    print('done')

    # class 2
    print('copying class 2 files ...')
    class2_srcdir = os.path.join(os.getcwd(), 'train_src', 'cases', 'class2')
    class2_tgtdir = os.path.join(os.getcwd(), 'train_src', '2')
    class2_csv = class2_srcdir + '.csv'
    copy_with_metadata(class2_srcdir, class2_tgtdir, '2', class2_csv)
    print('done')

    # class 3
    print('copying class 3 files ...')
    class3_srcdir = os.path.join(os.getcwd(), 'train_src', 'cases', 'class3')
    class3_tgtdir = os.path.join(os.getcwd(), 'train_src', '3')
    copy_all_cases(class3_srcdir, class3_tgtdir)
    print('done')

    # external test data
    # testroot = os.path.join(os.getcwd(), 'test_src', 'cases')
    # tgtroot = os.path.join(os.getcwd(), 'test_src')
    # csvfile = os.path.join(os.getcwd(), 'test_src', 'test_all.csv')
    # rearrange_data(testroot, tgtroot, csvfile)

