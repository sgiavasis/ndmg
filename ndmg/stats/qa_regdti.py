from scipy import ndimage
import os
import re
import numpy as np
import nibabel as nib
from matplotlib.colors import colorConverter
import matplotlib as mpl
from skimage.filters import threshold_otsu
mpl.use('Agg')  # very important above pyplot import
import matplotlib.pyplot as plt
import sys

def save_reg_pngs(path_to_fibs):
	plt.rcParams.update({'axes.labelsize': 'x-large', 'axes.titlesize':'x-large'})
        atlas_img = nib.load('/brainstore/MR/atlases/MNI152_template/MNI152_T1_1mm.nii.gz')
        atlas_data = atlas_img.get_data()
	x1 = 78 
	x2 = 90 
	x3 = 100
	y1 = 82
	y2 = 107
	y3 = 142
	z1 = 88
	z2 = 103
	z3 = 107
	for f in os.listdir(path_to_fibs):
		print(f)
		if (not re.match('^.*nii', f )): continue
		img = nib.load(path_to_fibs + f)
		data = img.get_data()
		cmap1 = mpl.cm.Greys_r
		# my_red_cmap.set_under(color="black", alpha="0")

		cmap2 = mpl.cm.hsv
		# cmap2.set_under(color="black", alpha="0")
		# my_blue_cmap.set_over(color="black", alpha="0")

		cmap2._init() # create the _lut array, with rgba values
		# create your alpha array and fill the colormap with them.
		# here it is progressive, but you can create whathever you want
		alphas = np.linspace(0, 0.8, cmap2.N+3)
		alphas = alphas * 0
		alphas = alphas + 0.5
		alphas[0] = 0
		cmap2._lut[:,-1] = alphas

		plt.show()	
		ax_x1 = plt.subplot(331)
		ax_x1.set_ylabel('Sagittal Slice: Y and Z fixed')
		ax_x1.set_title('X = ' + str(x1))
		ax_x1.yaxis.set_ticks([0, data.shape[2]/2, data.shape[2] - 1 ])
		ax_x1.xaxis.set_ticks([0, data.shape[1]/2, data.shape[1] - 1 ])
		image = data[x1,:,:,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(ndimage.rotate(atlas_data[x1,:,:], 90), interpolation='none', cmap=cmap1)
		plt.imshow(ndimage.rotate(data[x1,:,:,0], 90), interpolation='none', cmap=cmap2)
		ax_x2 = plt.subplot(332)
		ax_x2.set_title('X = ' + str(x2))
		ax_x2.get_yaxis().set_ticklabels([])
		ax_x2.get_xaxis().set_ticklabels([])
		image = data[x2,:,:,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(ndimage.rotate(atlas_data[x2,:,:], 90), interpolation='none', cmap=cmap1)
		plt.imshow(ndimage.rotate(data[x2,:,:,0], 90), interpolation='none', cmap=cmap2)
		ax_x3 = plt.subplot(333)
		ax_x3.set_title('X = ' + str(x3))
		ax_x3.get_yaxis().set_ticklabels([])
		ax_x3.get_xaxis().set_ticklabels([])
		image = data[x3,:,:,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(ndimage.rotate(atlas_data[x3,:,:], 90), interpolation='none', cmap=cmap1)
		plt.imshow(ndimage.rotate(data[x3,:,:, 0], 90), interpolation='none', cmap=cmap2)
		ax_y1 = plt.subplot(334)
		ax_y1.set_ylabel('Coronal Slice: X and Z fixed')
		ax_y1.set_title('Y = ' + str(y1))
		ax_y1.yaxis.set_ticks([0, data.shape[0]/2, data.shape[0] - 1 ])
		ax_y1.xaxis.set_ticks([0, data.shape[2]/2, data.shape[2] - 1 ])
		image = data[:,y1,:,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(ndimage.rotate(atlas_data[:,y1,:], 90), interpolation='none', cmap=cmap1)
		plt.imshow(ndimage.rotate(data[:,y1,:,0], 90), interpolation='none', cmap=cmap2)
		ax_y2 = plt.subplot(335)
		ax_y2.set_title('Y = ' + str(y2))
		ax_y2.get_yaxis().set_ticklabels([])
		ax_y2.get_xaxis().set_ticklabels([])
		image = data[:,y2,:,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(ndimage.rotate(atlas_data[:,y2,:], 90), interpolation='none', cmap=cmap1)
		plt.imshow(ndimage.rotate(data[:,y2,:,0], 90), interpolation='none', cmap=cmap2)
		ax_y3 = plt.subplot(336)
		ax_y3.set_title('Y = ' + str(y3))
		ax_y3.get_yaxis().set_ticklabels([])
		image = data[:,y3,:,0] 
		ax_y3.get_xaxis().set_ticklabels([])
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
                	plt.imshow(ndimage.rotate(atlas_data[:,y3,:], 90), interpolation='none', cmap=cmap1)
		plt.imshow(ndimage.rotate(data[:,y3,:,0], 90), interpolation='none', cmap=cmap2)
		ax_z1 = plt.subplot(337)
		ax_z1.set_ylabel('Axial Slice: X and Y fixed')
		ax_z1.set_title('Z = ' + str(z1))
		ax_z1.yaxis.set_ticks([0, data.shape[0]/2, data.shape[0] - 1 ])
		ax_z1.xaxis.set_ticks([0, data.shape[1]/2, data.shape[1] - 1 ])
		image = data[:,:,z1,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(atlas_data[:,:,z1], interpolation='none', cmap=cmap1)
		plt.imshow(data[:,:,z1, 0], interpolation='none', cmap=cmap2)
		ax_z2 = plt.subplot(338)
		ax_z2.set_title('Z = ' + str(z2))
		ax_z2.get_yaxis().set_ticklabels([])
		ax_z2.get_xaxis().set_ticklabels([])
		image = data[:,:,z2,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(atlas_data[:,:,z2], interpolation='none', cmap=cmap1)
		plt.imshow(data[:,:,z2,0], interpolation='none', cmap=cmap2)
		ax_z3 = plt.subplot(339)
		ax_z3.set_title('Z = ' + str(z3))
		ax_z3.get_yaxis().set_ticklabels([])
		ax_z3.get_xaxis().set_ticklabels([])
		image = data[:,:,z3,0] 
		thresh = threshold_otsu(image)
		binary = image < thresh
		image[ binary ] = 0
		plt.imshow(atlas_data[:,:,z3], interpolation='none', cmap=cmap1)
		plt.imshow(data[:,:,z3,0], interpolation='none', cmap=cmap2)
		fig = plt.gcf()
		fig.set_size_inches(12.5, 10.5, forward=True)
		plt.savefig('reg_pngs/' + f.split(".")[0] + '.png', format='png')
		print(f + " saved!")

if __name__ == '__main__':
	datasets = ['KKI2009', 'SWU4', 'HNU1', 'BNU1']
	base_path = '/brainstore/MR/data/'
	suffix = '/ndmg_v0033/reg_dti/'
	for d in datasets:
		save_reg_pngs(base_path + d + suffix)
		print('Done with ' + d)
