import os
import numpy as np
import cv2 as cv
data = '../Assignment_3_Data/Fingerprint'

def finger_data():
#	temp_num, n = 1500, 0
	new_dim = 200
	all_files = []
	for dirs in os.listdir(data):
		for sub_dir in os.listdir(data+'/'+dirs):
			print (data+'/'+dirs + '/' + sub_dir)
			for files in os.listdir(data+'/'+dirs+'/'+sub_dir):
#				n += 1
#				if n == temp_num:
#					break
				img = cv.imread(data+'/'+dirs+'/'+sub_dir+'/'+files) # , cv.IMREAD_GRAYSCALE
				# print(img.shape)
				img2 = cv.resize(img,(new_dim, new_dim))
				# stacked_img = np.stack((img,)*1, -1)
				all_files.append(img2)
	return all_files
