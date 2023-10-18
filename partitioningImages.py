import os

f = open('dataset/splits/dataset2/val.txt', 'w')

def matchingMask(imageName):
    return imageName[:-4] + '.png'

img_directory = 'Dataset2/Image/valid/'
msk_direcotry = 'Dataset2/Mask/valid/'

for files in os.listdir(img_directory):
    if files.endswith('.bmp'):
        f.write(img_directory + files + ' ' + msk_direcotry + matchingMask(files) + '\n')

f.close()