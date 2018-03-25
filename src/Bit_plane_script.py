from PIL import Image
import numpy as np

imagehide = Image.open('../imgs/Slider.jpg')
grayscale_image = imagehide.convert('L')
grayscale_image.save('graysliderpixels.png')
gray = np.array(grayscale_image)
gray.tofile('graysliders.txt', sep=" ", format="%s")

image = Image.open('../imgs/pixels.png')
mono = image.convert('1')
mono.save('monopixels.png')
mono = np.array(mono,dtype=np.uint8)
mono.tofile('monopixels.txt', sep=" ", format="%s")

gray_bit = np.unpackbits(gray).reshape((256, 256, 8))
gray_bit[:, :, 7] = mono
gray_bit = np.packbits(gray_bit).reshape((256, 256))
b = np.array(gray_bit)
b.tofile('gray_bit.txt', sep=" ", format="%s")
convert = Image.fromarray(b)
convert.save('end7.png')
