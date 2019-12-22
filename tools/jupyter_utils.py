import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math
from tqdm import tqdm
from scipy.cluster import hierarchy as hcy
from random import random
from scipy.spatial import distance as dis
import colorsys
from tqdm import tqdm


DISPLAY_SIZE = 6

class CompareItemBase():
    def __init__(self, path, display_name, prefix='', surfix=''):
        self.path = path
        self.display_name = display_name
        self.surfix = surfix
        self.prefix = prefix


def hide_ax_frame(ax):
        ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        ax.set_xticklabels([])
        plt.sca(ax)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)


def show_compare_images(compare_files, compare_item_bases):
    for file in compare_files:
        show_compare_image(file, compare_item_bases)


def show_compare_image(compare_file, compare_item_bases):
    fname, ext = os.path.splitext(os.path.basename(compare_file))

    imgPlot = ImageLinePlotter(0, len(compare_item_bases))
    for item_base in compare_item_bases:
        item_path = os.path.join(item_base.path, item_base.prefix + fname + item_base.surfix + ext)
        img = cv2.imread(item_path)
        if img is None:
            print('Image is None : ', item_path)
            continue
        imgPlot.add_image(img, title=item_base.display_name)
    imgPlot.show_plot()



def show_compare_images_old(original_file_paths, compare_item_bases, original_display_name='origin'
                                                 , hide_frame=True, padding_item=0, row_data_count=1, with_data_number=True):
    if len(original_file_paths) < 1:
        print('compare data is Nothing .')
        return
    if len(compare_item_bases) < 1:
        print('compare path is Nothing .')

    compare_item_bases.insert(0, CompareItemBase(os.path.dirname(original_file_paths[0]), original_display_name))

    for k, origin_path in enumerate(original_file_paths):
        print(os.path.basename(origin_path))
        base_name, ext = os.path.splitext(os.path.basename(origin_path))

        compare_item_count = len(compare_item_bases) + padding_item
        if k % row_data_count == 0:
            fig = plt.figure(k, figsize=(DISPLAY_SIZE * compare_item_count * row_data_count, DISPLAY_SIZE))

        axes = []
        for j, cbi in enumerate(compare_item_bases):
            ax = fig.add_subplot(1, compare_item_count * row_data_count, j + 1 + (k % row_data_count * compare_item_count))
            if with_data_number:
                if cbi.display_name == '':
                    plt.title('[%d]' % k)
                else:
                    plt.title('[%d] : %s' % (k, cbi.display_name))
            else:
                plt.title(cbi.display_name)
            if hide_frame:
                hide_ax_frame(ax)
            axes.append(ax)

        for i, (ax, cbi) in enumerate(zip(axes, compare_item_bases)):
            item_path = os.path.join(cbi.path, base_name + cbi.surfix + ext)
            if os.path.exists(item_path):
                img = cv2.cvtColor(cv2.imread(item_path), cv2.COLOR_BGR2RGB)
                fig.sca(ax)
                plt.imshow(img)
            else:
                print('not exists : ', item_path)

        if (k + 1) % row_data_count == 0:
            plt.show()


def show_img_list(files, col_item_count=5, disp_size=3):
    for k, file in enumerate(files):
        c = k + 1
        ax_id = c % col_item_count
        if ax_id == 1:
            fig = plt.figure(figsize=(disp_size * col_item_count, disp_size))
        if ax_id == 0:
            ax_id = col_item_count

        ax = fig.add_subplot(1, col_item_count, ax_id)
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(os.path.basename(file))
        hide_ax_frame(ax)
        if ax_id == col_item_count:
            plt.show()


class ImageLinePlotter():
    def __init__(self, fig_id, plot_area_num=6, display_size=DISPLAY_SIZE):
        self.fig_id = fig_id
        self.display_size = display_size
        self.plot_area_num = plot_area_num
        self.figsize = (self.plot_area_num*self.display_size, self.display_size)
        self.fig = plt.figure(self.fig_id, figsize=self.figsize)
        self.image_infos = [None] * self.plot_area_num
        self.image_count = 0


    def add_image(self, img, title='', pos=None):
        if pos is None:
            pos = len(self.image_infos) - 1
            for k, img_info in enumerate(self.image_infos):
                if img_info is None:
                    pos = k+1
                    break

        self.image_infos[pos-1] = (img, title)


    def show_plot(self):
        for k, img_info in enumerate(self.image_infos):
            if img_info is None:
                continue
            (img, title) = img_info
            ax = self.fig.add_subplot(1, self.plot_area_num, k + 1)
            plt.title(title)
            hide_ax_frame(ax)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

"""
# plot images
img_ploter = ImageLinePlotter(0, plot_area_num=4, display_size=5)

img_ploter.add_images(img1, title='image A')
img_ploter.add_images(img2, title='image B', pos=3)
img_ploter.add_images(img3, title='image C', pos=4)

img_ploter.show_plot()
"""
