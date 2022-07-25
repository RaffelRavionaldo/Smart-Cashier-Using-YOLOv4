import numpy as np
import cv2
import os
import glob
import argparse

from helpers import *

class yoloRotatebbox:
    def __init__(self, filename, image_ext, angle):
        #assert os.path.isfile(filename + image_ext)
        #assert os.path.isfile(filename + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image().shape[:2]
        f = open(self.filename + '.txt', 'r')
        f1 = f.readlines()
        new_bbox = []
        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                #inisialisasi titik koordinat bounding box pada gambar sebelum rotasi
                (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)

                #menggeser titik sudut bounding box
                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                #menentukan titik koordinat bounding box baru
                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)
                #memasukkan nilai titik koordinat bounding box baru ke variabel new_bbox
                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                 new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # Untuk mendapatkan lebar dan panjang
                                              # dari gambar sebelum dirotasi
        image_center = (width / 2,
                        height / 2)

        # fungsi getRotationMatrix2D digunakan untuk membuat
        # transformasi matriks (2x3) yang akan digunakan untuk merotasi gambar
        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # menghitung nilai absolut cos dan sin dari transformasi matriks
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # menghitung height dan width dari batas gambar untuk rotasi
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # kurangi bagian tengah gambar sebelum rotasi (kembalikan gambar ke aslinya)
        # dan tambahkan koordinat pusat gambar setelah dirotasi
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # Putar gambar dengan batas baru
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat

if __name__ == "__main__":

    sudut = [0,45,90,270]
    for angle in sudut:
        print(angle)
        os.chdir(r"lokasi folder gambar yang akan di augmentasi")
        for file in glob.glob("*.jpg"):
            image_name = file.split('.')[0]         #untuk mengambil data gambar
            image_ext = '.'+file.split('.')[1]      #untuk mengambil file txt yang berisi titik
                                                    #koordinat dan kode produk

            # initiate the class
            im = yoloRotatebbox(image_name, image_ext, angle)

            bbox = im.rotateYolobbox()
            image = im.rotate_image()

            cv2.imwrite(os.path.join(r"E:\dataset\test", 'rotated_'+ image_name +'_' + str(angle) + '.jpg'), image)

            file_name = os.path.join(r"E:\dataset\test",'rotated_'+image_name+'_' + str(angle) + '.txt')
            if os.path.exists(file_name):
                os.remove(file_name)

            # to write the new rotated bboxes to file
            for i in bbox:
                with open(file_name, 'a') as fout:
                    fout.writelines(
                        ' '.join(map(str, cvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')