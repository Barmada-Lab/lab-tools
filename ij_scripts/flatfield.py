#@ File background (label='Background',		description='Background image to subtract from images',  style='file')
#@ File images 	   (label='Base directory', description='The base directory of images to flatfield', style='directory')
#@ File output     (label='Output',			description='Output directory for flatfielded images',	 style='directory')
#@ ConvertService convertService
#@ DatasetIOService ds
#@ DatasetService dds 
#@ UIService ui
#@ ImageJ ij

from glob import glob

from io.scif.img import ImgSaver

import os

background = background.toString()
images = images.toString()
output = output.toString()

background_img = ij.io().open(background)

saver = ImgSaver()
for root, dirs, files in os.walk(images):
	for file_name in files:
		if not file_name.endswith(".tif"):
			continue
			
		input_path = os.path.join(root, file_name)
		output_path = os.path.join(output, os.path.relpath(input_path, images))
		
		pardir = os.path.abspath(os.path.join(output_path, os.path.pardir))
		if not os.path.isdir(pardir):
			os.makedirs(pardir)
			
		img = ij.io().open(input_path)
		
		converted = ij.op().convert().float32(img)
		subtracted = ij.op().create().imgPlus(converted)
		ij.op().math().subtract(subtracted, converted, background_img)
	
		saver.saveImg(output_path, subtracted)
		print(output_path)