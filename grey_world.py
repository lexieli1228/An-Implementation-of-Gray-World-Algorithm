import numpy as np
import cv2
import os
import math

def grey_world_0(source_path='./test_picture/gracie_gold_0.png', target_path='./output_picture/round_0/', option_x = 0):
    img = cv2.imread(source_path)
    img_1 = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    shape = img.shape
    average_channel_0 = np.average(img_1[:, :, 0])
    average_channel_1 = np.average(img_1[:, :, 1])
    average_channel_2 = np.average(img_1[:, :, 2])

    average_channels = 128

    if option_x == 0:
        average_channels = (average_channel_0 + average_channel_1 + average_channel_2) / 3

    img_output = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            img_output[i][j][2] = ((average_channels / average_channel_0) * img_1[i][j][0])
            img_output[i][j][1] = ((average_channels / average_channel_1) * img_1[i][j][1])
            img_output[i][j][0] = ((average_channels / average_channel_2) * img_1[i][j][2])

    img_output_final = np.clip(img_output, 0, 255).astype('uint8')

    path_new_0 = source_path.split('/')
    path_new_1 = path_new_0[2].split('.')
    target_path_temp = target_path + path_new_1[0] + '_modified.' + path_new_1[1]
    cv2.imwrite(target_path_temp, img_output_final)
    print("output picture successful at {}".format(target_path_temp))

def grey_world_1(source_path='./test_picture/gracie_gold_0.png', target_path='./output_picture/round_0/', division_0 = 3, division_1 = 3, option_x = 0):
    img = cv2.imread(source_path)
    img_1 = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    shape = img.shape

    average_matrix = np.zeros((division_0 + 1, division_1 + 1, 4)) # for every block, calculate the info
    stride_0 = math.ceil(shape[0]/division_0)
    stride_1 = math.ceil(shape[1]/division_1)

    if option_x == 0:
        for i in range(division_0):
            for j in range(division_1):
                boundary_i = min(stride_0*(i + 1) - 1, shape[0])
                boundary_j = min(stride_1*(j + 1) - 1, shape[1])
                average_matrix[i][j][0] = np.average(img_1[stride_0*i:boundary_i, stride_1*j:boundary_j, 0])
                average_matrix[i][j][1] = np.average(img_1[stride_0*i:boundary_i, stride_1*j:boundary_j, 1])
                average_matrix[i][j][2] = np.average(img_1[stride_0*i:boundary_i, stride_1*j:boundary_j, 2])
                average_matrix[i][j][3] = (average_matrix[i][j][0] + average_matrix[i][j][1] + average_matrix[i][j][2]) / 3

    else:
        for i in range(division_0):
            for j in range(division_1):
                boundary_i = min(stride_0*(i + 1) - 1, shape[0])
                boundary_j = min(stride_1*(j + 1) - 1, shape[1])
                average_matrix[i][j][0] = np.average(img_1[stride_0*i:boundary_i, stride_1*j:boundary_j, 0])
                average_matrix[i][j][1] = np.average(img_1[stride_0*i:boundary_i, stride_1*j:boundary_j, 1])
                average_matrix[i][j][2] = np.average(img_1[stride_0*i:boundary_i, stride_1*j:boundary_j, 2])
                average_matrix[i][j][3] = 128

    img_output = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            index_i = math.floor(i / stride_0)
            index_j = math.floor(j / stride_1)
            img_output[i][j][2] = ((average_matrix[index_i][index_j][3] / average_matrix[index_i][index_j][0]) * img_1[i][j][0])
            img_output[i][j][1] = ((average_matrix[index_i][index_j][3] / average_matrix[index_i][index_j][1]) * img_1[i][j][1])
            img_output[i][j][0] = ((average_matrix[index_i][index_j][3] / average_matrix[index_i][index_j][2]) * img_1[i][j][2])

    img_output_final = np.clip(img_output, 0, 255).astype('uint8')

    path_new_0 = source_path.split('/')
    path_new_1 = path_new_0[2].split('.')
    target_path_temp = target_path + path_new_1[0] + '_modified.' + path_new_1[1]
    cv2.imwrite(target_path_temp, img_output_final)
    print("output picture successful at {}".format(target_path_temp))

source_dir_name = './test_picture/'
target_dir_name = './output_picture/grey_world_1/round_0_210_1/'

picture_list = os.listdir(source_dir_name)
print(picture_list)

for i in range(len(picture_list)):
    source_p = source_dir_name + picture_list[i]
    target_p = target_dir_name
    grey_world_1(source_p, target_p, 3, 3, 1)

print("conversion successful")


