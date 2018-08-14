#!/usr/bin/env python3

import time
import extension
import cv2
import numpy as np



def rescale_frame(frame, percent=75):
    '''rescale image for imshow'''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def main():

    src_img = cv2.imread("./pano_4K.jpg")
    
    start_time = time.time()
    arr = [src_img[:,:, 0], src_img[:, :, 1], src_img[:, :, 2]]
    dst_img = np.array(extension.get_image(arr), copy=False)
    # print(type(dst_img))

    end_time = time.time()
    print(end_time - start_time)

    cv2.imshow("dst", dst_img)
    cv2.waitKey()


    


if __name__=="__main__":
    main()