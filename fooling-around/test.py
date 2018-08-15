#!/usr/bin/env python3

import time
import extension
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rescale_frame(frame, percent=75):
    '''rescale image for imshow'''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def main():

    src_img = cv2.imread("./pano_4K.jpg")

    times = []

    for i in range(0, 1000):
        start_time = time.time()
        arr = [src_img[:,:, 0], src_img[:, :, 1], src_img[:, :, 2]]
        dst_img = np.array(extension.get_image(arr, [0, 0, 0]), copy=False)
        # print(type(dst_img))
        end_time = time.time()
        diff = end_time - start_time
        times.append(diff)


    print(sum(times) / len(times))

    x_axis = [ i for i in range(len(times)) ]
    plt.plot(x_axis, times)
    plt.savefig('gpu.png')

if __name__=="__main__":
    main()