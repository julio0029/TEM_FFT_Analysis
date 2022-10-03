#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
-------------------------------------------------------------------------------
Copyright© 2021 Jules Devaux / Cody Williams. All Rights Reserved
Open Source script under Apache License 2.0
-------------------------------------------------------------------------------

Does FFT and other analysis to TEM images

Requires gray_scale images with mitochondria only (binary mask already applied by user)
DATA>Temp>State

After extraction, each image is exported as
dict:{filename, temperature, state, image, magnification, fft}

'''
import os, pickle
import numpy as np
import pandas as pd
import pingouin
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from sklearn.cluster import MeanShift
from scipy.fft import fft, ifft
from scipy import ndimage
from scipy.signal import find_peaks
import cv2


current_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH=f"{current_path}/DATA/"
STATES=['Intact', 'Permeabilised']
TEMPERATURES=[18,24,26,28,30]
WRASSES=[20,21,22,23]


def get_fish_nb(filename):

	if 'Copy' in filename:
		fish=filename.split(' -')[0][-6:-4]
	else: fish=filename[-6:-4]

	return int(fish)



def extract_magnification(filename):
	'''
	Exctact magnification from filename
	'''
	magnification=None
	try:
		_splits=filename.split('-')
		for _mag in ['m','sa','M','SA']:
			_select=[i for i in _splits if _mag in i]
			if _select:
				magnification=_select[0].split(_mag)[1]
	except Exception as e:
		print(f"ERROR: extract_magnification: {e}")

	if magnification is None:
		print(f"ERROR: {filename} no magnification found")

	return magnification



def exctract_files(DATA_PATH):
	'''
	Extract images and meta.
	Returns a list of dictionary as:
	below
	'''
	images=[]	# List to store image dictionaries
	for temperature in TEMPERATURES:
		for state in STATES:
			#files=[file for file in os.listdir(f"{DATA_PATH}{temperature} degree/{state}/") if '.tif' in file]
			try:
				for filename in os.listdir(f"{DATA_PATH}{temperature} degrees/{state}/"):
					_path=f"{DATA_PATH}{temperature} degrees/{state}/{filename}"
					if '.tif' in filename:
						print(f"Exctacting {filename}...")
						fish=get_fish_nb(filename)
						magnification=extract_magnification(str(filename))
						image=imread(_path)
						images.append({
							'filename':str(filename),
							'fish':fish,
							'temperature':int(temperature),
							'state':str(state),
							'magnification':magnification,
							'image':image,
							})
			except:pass
	return images



def linearise_fft(fft2D, window=10, magnification=1):
	'''
	Returns 1D x axis middle to right from fft graph 
	'''
	_middle=int(len(fft2D)/2)
	_select=fft2D[_middle+1][(_middle+1):]
	df=pd.DataFrame(_select).rolling(window=window).mean()
	df.index=[i/(magnification/1000) for i in df.index]

	return df



def do_fft(image, _graph=False):
	'''
	Return 2D numpy array: X=pixel, Y=FFT
	'''

	image_fourier = np.fft.fftshift(np.fft.fft2(image))


	_select=linearise_fft(image_fourier)


	if _graph:
		fig, ax=plt.subplots(3)
		ax[1].imshow(np.log(abs(image_fourier)),cmap='gray')
		ax[0].imshow(image, cmap='gray')
		ax[2].plot(_select)
		plt.show()

	return _select


def count_membranes(image, threshold=176):

	image=np.where(image<threshold,1,0)
	_membranes=np.count_nonzero(image<50)/len(image)
	return _membranes



def pixel_density(image):
	image=np.where(image==85,float('nan'),image)
	hist, _bin = np.histogram(image,256,[0,255])
	return hist



def normalise(image):
	'''
	threshold depending on bin grayscale profile.
	Ideally bckg - fluid - membranes
	'''
	
	return image-image.min()



def sandbox_average(images, _methods=[0]):

	print("Doing averages...")

	# Retrieve magnification range
	magnifications=[i['magnification'] for i in images]
	df=pd.DataFrame(magnifications).astype('int')
	

	for _state in STATES:
		for temperature in TEMPERATURES:

			all_fft=[]

			for i in images:
				if (i['temperature']==temperature) and (i['state']==_state):
					for _method in _methods:
						if _method==1:
							coef=1/((int(i['magnification'])-int(df.iloc[:,0].min()))/int((df.iloc[:,0].min()))+1)
						elif _method==2:
							coef= int(i['magnification'])/(int(i['magnification'])/int(df.iloc[:,0].max()))
						elif _method==0:
							coef=1
						new_fft=i['fft']
						new_fft.index=new_fft.index*coef

						# plt.clf()
						# plt.plot(new_fft)
						# plt.title(f"{_state} {temperature}")
						# plt.savefig(f"{_state}-{temperature}-{_method}.pdf")


						all_fft.append(new_fft)


			# average all FFTs:
			all_fft=pd.concat(all_fft, axis=1)
			all_fft=all_fft.sum(axis=1)

			plt.clf()

			plt.plot(all_fft)
			plt.title(f"{_state} {temperature}")
			plt.savefig(f"{_state}-{temperature}-{_method}.pdf")



def normalise_all_image_test():

	images=exctract_files(DATA_PATH)


	# create a summary excel spreadsheet


	# temporary list of ffts
	all_fft=[]

	for file in images:
		image=file['image']

		#image=np.where(image<200,1,0)
		plt.imshow(image, cmap='gray')
		plt.show()
		

		normalised_image=pixel_density(image)
		plt.plot(normalised_image)
		plt.show()
		exit()

		_membranes=np.count_nonzero(image<50)/len(image)
		print(_membranes)
		exit()
		image_fourier = np.fft.fftshift(np.fft.fft2(image))
		#image_fourier=fft(image)

		_middle=int(len(image_fourier)/2)
		_select=image_fourier[_middle][(_middle+1):]
		_select=pd.DataFrame(_select).rolling(window=20).mean()

		all_fft.append(_select)
		#plt.plot(_select)

	# average all FFTs:
	all_fft=pd.concat(all_fft, axis=1)
	all_fft=all_fft.sum(axis=1)


	# ==== GRAPH ====
	#fig, ax=plt.subplots(2)

	#ax[0].imshow(np.log(abs(IMAGES[0])), cmap='gray')
	plt.plot(all_fft)
	plt.show()


def sandbox_rotation():
	# First extract files. Easier to process later
	files=exctract_files(DATA_PATH)

	data=[]
	for file in files:
		# define h and w of image
		(h,w)=file['image'].shape[:2]
		centre=(w/2, h/2)
		scale=1
		for i in range(0,360,30):
			M=cv2.getRotationMatrix2D(centre, i, scale)
			rotated=cv2.warpAffine(file['image'], M, (w,h), borderValue=(85,85,85))
			_fft=do_fft(rotated, _graph=True)
			data.append(rotated)
	print(data)
		


def graph_ffts(_mode='all'):
	'''
	Ugly, will have to re-write
	'''


	with open("FFTS.pkl",'rb') as f:
		FFTS=pickle.load(f)


	#temporary storage to run mean across wrasses
	_TEMP={s:{} for s in STATES}
	for s in STATES:
		_TEMP[s].update({t:[] for t in TEMPERATURES})

	COLORS=['#ffca00', '#ffb30f', '#f95502', '#ff2900', '#d12200']

	fig, ax = plt.subplots(len(STATES), len(TEMPERATURES))
	for s in range(len(STATES)):
		for t in range(len(TEMPERATURES)):
			_datatemp=[]
			for w in range(len(WRASSES)):
				try:
					freqs=pd.concat(FFTS[STATES[s]][TEMPERATURES[t]][WRASSES[w]], axis=1)

					_sum=freqs.sum(axis=1)
					_mean=freqs.mean(axis=1)
					_sem=freqs.sem(axis=1)

					_datatemp.append(_mean)
					
					if 'all' in _mode:
						ax[s,t].set_title(f"{STATES[s]} - {TEMPERATURES[t]}ºC", fontsize=10)
						#ax[s,t].set_ylim(0,100000)

						for c in freqs.columns:
							#_logfreq=np.log10(freqs[c])
							peaks,_=find_peaks(freqs[c])
							ax[s,t].plot(freqs[c])
							#ax[s,t].plot(freqs[c].iloc[peaks].index, freqs[c].iloc[peaks].values, 'xr', lw=5)

					elif 'fish' in _mode:
						ax[s,t].plot(_mean)
						if 'sem' in _mode:
							ax[s,t].plot((_mean+_sem), '--', c='black')
							ax[s,t].plot((_mean-_sem), '--', c='black')
						# Plot peaks
						peaks,_=find_peaks(_mean)
						ax[s,t].plot(_mean.iloc[peaks].index, _mean.iloc[peaks].values, 'or', lw=20)

				except Exception as e:print(e)

			if 'mean' in _mode:
				try:
					df=pd.concat(_datatemp, axis=1)
					df.index=df.index/10000
					_mean=df.mean(axis=1)
					_sem=df.sem(axis=1)


					ax[s,t].plot(_mean, lw=10, c=COLORS[t])
					if 'sem' in _mode:
						ax[s,t].plot((_mean+_sem), '--', c='black')
						ax[s,t].plot((_mean-_sem), '--', c='black')
					# Plot peaks
					peaks,_=find_peaks(_mean)
					ax[s,t].plot(_mean.iloc[peaks].index, _mean.iloc[peaks].values, 'og', lw=10)
				except Exception as e:print(e)
			_datatemp=[]


			# Set titles etc
			ax[s,t].set_title(f"{STATES[s]} - {TEMPERATURES[t]}ºC", fontsize=10)
			if t==0:
				ax[s,t].set_ylabel("Pixel density")
			elif t!=0:
				ax[s,t].set_yticklabels([])
			if s==1 and t==2:
				ax[s,t].set_xlabel("Distance (nm)")
			ax[s,t].set_ylim(0,30000)


	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()

	plt.subplots_adjust(wspace=0)
	plt.savefig('FFT.pdf')
	plt.show()


def sandbox():
	with open("FFTS.pkl",'rb') as f:
		FFTS=pickle.load(f)

	PEAKS={s:{} for s in STATES}
	for s in STATES: PEAKS[s].update({t:[] for t in TEMPERATURES})

	#fig, ax = plt.subplots(len(STATES), len(TEMPERATURES))
	# Find peaks for each file and append to dict
	for s in range(len(STATES)):
		for t in range(len(TEMPERATURES)):

			freqs=pd.concat(FFTS[STATES[s]][TEMPERATURES[t]], axis=1)
			for c in freqs.columns:
				peaks,_=find_peaks(freqs[c])
				PEAKS[STATES[s]][TEMPERATURES[t]].append(
						pd.DataFrame(freqs[c].iloc[peaks].values,
								index=freqs[c].iloc[peaks].index))
	
	# Average and sem peaks for each stuff
	fig, ax = plt.subplots(len(STATES), len(TEMPERATURES))
	for s in range(len(STATES)):
		for t in range(len(TEMPERATURES)):
			dfs=PEAKS[STATES[s]][TEMPERATURES[t]]
			df=pd.concat(dfs, axis=0, ignore_index=False).sort_index()
			df.index=df.index/1000
			df=df.reset_index().set_index(0)#.fillna(0)

			# Average the number of peak found per sample
			_average_peak_nb=int(np.mean([len(df) for df in dfs]))

			model=MeanShift(bin_seeding=True)
			#model.fit(df)
			yhat = model.fit_predict(df)
			# retrieve unique clusters
			clusters = np.unique(yhat)
			colors=['r','b','g']
			for i in range(len(clusters)):
				# get row indexes for samples with this cluster
				row_ix = np.where(yhat == clusters[i])
				# create scatter of these samples
				ax[s,t].scatter(df.iloc[row_ix].values, df.iloc[row_ix].index, c=colors[i])


			#ax[s,t].scatter(df.index.to_list(), df.iloc[:,0].to_list())
			ax[s,t].set_title(f"{STATES[s]} - {TEMPERATURES[t]}ºC", fontsize=10)
			ax[s,t].set_ylim(0,100000)
			if t==0:
				ax[s,t].set_ylabel("Pixel density")
			elif t!=0:
				ax[s,t].set_yticklabels([])
			if s==1 and t==2:
				ax[s,t].set_xlabel("Distance (nm)")

			
	plt.subplots_adjust(wspace=0)
	plt.show()
		





def main():
	_FFTS={s:{} for s in STATES}
	for s in STATES:
		_FFTS[s].update({t:{} for t in TEMPERATURES})
		for t in TEMPERATURES:
			_FFTS[s][t].update({w:[] for w in WRASSES})

	# First extract files. Easier to process later
	files=exctract_files(DATA_PATH)


	for file in files:
		img=file['image']
		_fft=np.log(abs(np.fft.fftshift(np.fft.fft2(img)).real))#.real
		
		# Creat graph and save as image.
		plt.imshow(_fft, cmap='gray')
		plt.axis('off')
		_tempFTT=f"/Users/julesdevaux/Desktop/{file['filename']}_fft.tif"
		plt.savefig(_tempFTT, bbox_inches='tight', pad_inches=0, transparent=True)
		plt.close()

		# Load the pixelised fft
		_fft=cv2.imread(_tempFTT,0)
		os.remove(_tempFTT)
		_fft=cv2.medianBlur(_fft,5)

		# Thresholding
		_fftBin=cv2.adaptiveThreshold(_fft, 255,
							cv2.ADAPTIVE_THRESH_MEAN_C,
							cv2.THRESH_BINARY,25,2)
		# Invert
		_fftBin=cv2.bitwise_not(_fftBin)

		# Sum rotative 1D (along x) pixel density.
		(h,w)=_fftBin.shape[:2]
		centre=(w/2, h/2)
		scale=1

		freqs=[]
		for i in range(0,360,1):
			M=cv2.getRotationMatrix2D(centre, i, scale)
			rotated=cv2.warpAffine(_fftBin, M, (w,h)) #, borderValue=(85,85,85)
			_freq=linearise_fft(rotated)
			freqs.append(_freq)

			#plt.imshow(rotated, cmap='gray')
			#plt.show()
		_freq=pd.concat(freqs, axis=1).T
		_freq=_freq.sum(axis=0)


		_FFTS[file['state']][file['temperature']][file['fish']].append(_freq)


		# Graph
		# fig, ax = plt.subplots(4)
		# fig.suptitle(f"{file['temperature']}ºC - {file['state']}")
		# ax[0].imshow(img, cmap='gray')
		# ax[1].imshow(_fft, cmap='gray')
		# ax[2].imshow(_fftBin, cmap='gray')
		# ax[3].plot(_freq)
		# plt.show()


		print(f"Done {file['filename']}")
	with open('FFTS.pkl','wb') as f:
		pickle.dump(_FFTS,f)


if __name__=='__main__':
	#main()
	#extract_magnification('W11.1-040622-sa6500-18.tif')
	graph_ffts(_mode='mean_sem')


