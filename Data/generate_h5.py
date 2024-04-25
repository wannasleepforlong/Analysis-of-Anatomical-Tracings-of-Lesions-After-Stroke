import os
import nibabel as nib
import numpy as np

import csv
from tqdm import tqdm
import h5py
import argparse

import random


class CSV:  # Done
    def __init__(self, path, del_rows=None, del_cols=None):
        if del_cols is None:
            del_cols = []
        if del_rows is None:
            del_rows = [0]
        self.lines_csv = self.read_csv(path, del_rows, del_cols)

    def read_csv(self, path, del_rows=None, del_cols=None):
        if del_cols is None:
            del_cols = [0]
        if del_rows is None:
            del_rows = [0]
        csv_path = path
        with open(csv_path, 'r') as f_csv:
            csv_reader = csv.reader(f_csv)
            lines_csv = np.array([line for line in csv_reader])

            if del_rows is not None:
                lines_csv = np.delete(lines_csv, del_rows, axis=0)

            if del_cols is not None:
                lines_csv = np.delete(lines_csv, del_cols, axis=1)

        return lines_csv

    def get_rows(self, rows):
        return [self.lines_csv[line] for line in rows]

    def get_cols(self, cols):
        return np.transpose([self.lines_csv[:, line] for line in cols])


def load_nii(file):  # Done
    img = nib.load(file).get_fdata()
    return img


# def read_gz_file(path):
#     if os.path.exists(path):
#         with gzip.open(path, 'r') as pf:
#             return pf
#     else:
#         print('the path [{}] is not exist!'.format(path))
#
#
# def show_img(tem):
#     if type(tem).__name__ == 'str':
#         img = load_nii(tem)
#         img = img.transpose((1, 0, 2))
#     else:
#         img = tem
#
#     print(np.max(img))
#
#     img = img / np.max(img) * 255
#     img = (np.array(img)).astype('uint8')
#
#     for i in range(len(img)):
#         if np.sum(img[i]) != 0:
#             print()
#             fig = plt.figure()
#
#             ax = fig.add_subplot(111)
#             im = ax.imshow(img[i], 'gray')
#             plt.title(i)
#             plt.colorbar(im)
#             plt.show()


def get_data(num=None,
             dataset_path='E:\\data\\ATLAS_R1.1'):
    if num is None:
        num = [0, 654]
    path_csv = os.path.join(dataset_path, '20220425_ATLAS_2.0_MetaData.csv')
    csv_reader = CSV(path_csv)
    # lines_csv = csv_reader.lines_csv
    deface = []
    tem_deface = []
    tem_seg = []
    seg = []
    count = 0

    for file in tqdm(csv_reader.get_cols([0, 1, 23])):
        if count < num[0]:
            count += 1
            continue
        if count > num[1]:
            break
        nii_path = os.path.join(dataset_path, "Training", file[-1], file[0], file[1], "anat")

        for root, dirs, files in os.walk(nii_path):
            for inside_file in files:
                check_file = str(inside_file).split('_')

                if check_file[-1][0] == 'T':
                    tem_deface = load_nii(os.path.join(nii_path, inside_file)).transpose((2, 1, 0)) / 100

                elif check_file[-1][:4] == 'mask':
                    tem = load_nii(os.path.join(nii_path, inside_file)).transpose((2, 1, 0)).astype(np.int64)
                    tem[tem > 0] = 1
                    tem_seg.append(tem)

        deface.append(tem_deface)
        tem_seg = np.sum(tem_seg, axis=0)
        tem_seg[tem_seg > 1] = 1
        seg.append(tem_seg)
        tem_deface = []
        tem_seg = []
        count += 1

    # deface = np.array(deface)
    # seg = np.array(seg)
    # deface /= 100
    return deface, seg


def to_slice(deface, seg, model=None, num_slices=189):
    deface_slice = []
    seg_slice = []
    pos_deface = []
    neg_deface = []
    pos_seg = []
    neg_seg = []
    for c1, i in enumerate(seg):
        for c2, k in enumerate(i):
            if c2 % 3 == 0:
                # This should give us a random sample of 50 slices thus reducing data for individual
                if model == 'pos':
                    if np.sum(k) != 0:
                        deface_slice.append(deface[c1][c2])
                        seg_slice.append(seg[c1][c2])
                elif model == 'all':
                    deface_slice.append(deface[c1][c2])
                    seg_slice.append(seg[c1][c2])
                else:
                    if np.sum(k) != 0:
                        pos_deface.append(deface[c1][c2])
                        pos_seg.append(seg[c1][c2])
                    else:
                        neg_deface.append(deface[c1][c2])
                        neg_seg.append(seg[c1][c2])
    if model != 'pos' and model != 'all':
        index = np.arange(len(neg_deface))
        np.random.shuffle(index)
        neg_deface = np.array(neg_deface)[index]
        neg_seg = np.array(neg_seg)[index]

        if model[0] / model[1] < len(pos_deface) / len(neg_deface):
            print('error', model[1] / model[0], '<', len(pos_deface) / len(neg_deface))
            return 0
        else:
            pos_num = len(pos_deface)
            seg_num = int(model[1] / model[0] * pos_num)
            deface_slice = np.zeros((pos_num + seg_num, deface[0].shape[1], deface[0].shape[2]))
            seg_slice = np.zeros((pos_num + seg_num, deface[0].shape[1], deface[0].shape[2]))
            deface_slice[:pos_num] = pos_deface
            deface_slice[pos_num:] = neg_deface[:seg_num]
            seg_slice[:pos_num] = pos_seg
            seg_slice[pos_num:] = neg_seg[:seg_num]
            print('pos_num:', pos_num, 'seg_num:', seg_num)
    return np.array(deface_slice), np.array(seg_slice)


def train_data_generator(dataset_path='E:\\data\\ATLAS_R1.1'):
    h5_path = 'ATLAS.h5'
    if not os.path.exists(h5_path):
        deface, seg = get_data([0, 199], dataset_path=dataset_path)
        deface = np.array(deface)
        deface_slice_train, seg_slice_train = to_slice(deface[:], seg[:], 'all', num_slices=50)

        print('generating h5 file for ATLAS dataset')
        with h5py.File('train200-50.h5', 'w') as file_train:
            file_train.create_dataset('data', data=deface_slice_train)
            file_train.create_dataset('label', data=seg_slice_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='ATLAS_R2.0/ATLAS_2', type=str,
                        help='path of ATLAS_R2.0')
    args = parser.parse_args()
    train_data_generator(dataset_path=args.dataset_path)
