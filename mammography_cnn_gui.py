import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import numpy as np
import cv2
import math
import tensorflow as tf

class mammography_cnn(tk.Frame):
	def __init__(self, parent):
		tk.Frame.__init__(self, parent)
		self.model_name = 'inceptionv3'
		self.model = tf.keras.models.load_model('models/' + self.model_name + '.h5')
		self.model = tf.keras.models.Model(inputs = self.model.input, outputs = (self.model.layers[-3].output, self.model.layers[-1].output))
		self.img_panel = None
		self.image_orig = None
		self.heatmap = None
		self.heatmap_toggle = 0
		btnbrowse = tk.Button(text="Browse", command=self.browse)
		btnbrowse.grid(row=1, column=0, columnspan = 4, sticky="NSEW", padx=5, pady=5)
		
	def browse(self):
		path = filedialog.askopenfilename()
		if len(path) > 0:
			self.update_panel(path)

	def update_panel(self, path):
		self.image_orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		self.image_orig = self.otsu_segmentation(self.image_orig)
		image = cv2.resize(self.image_orig, (int(self.image_orig.shape[1]//7), int(self.image_orig.shape[0]//7)))
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		malignant = tk.Label(text=path)
		malignant.grid(row=2, column=0, columnspan = 4, sticky="NSEW", padx=5, pady=5)
		if self.img_panel is None:
			self.img_panel = tk.Label(image=image)
			self.img_panel.image = image
			self.img_panel.grid(row = 0, column = 0, columnspan = 4, sticky="NSEW")
		else:
			self.img_panel.configure(image=image)
			self.img_panel.image = image
		self.classify()
		malignant = tk.Label(text="Malignant confidence: " + str(self.pred))
		malignant.grid(row=3, column=0, columnspan = 4, sticky="NSEW", padx=5, pady=5)
		btnheatmap = tk.Button(text="Toggle heatmap", command=self.toggle_cam)
		btnheatmap.grid(row=4, column=0, columnspan = 4, sticky="NSEW", padx=5, pady=5)

	def otsu_segmentation(self, image):
	    otsu = cv2.threshold(image, 0, 1, cv2.THRESH_OTSU)[1]
	    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	    contours_areas = [cv2.contourArea(cont) for cont in contours]
	    biggest_contour_idx = np.argmax(contours_areas)
	    tx, ty, tw, th = cv2.boundingRect(contours[biggest_contour_idx])
	    return image[ty:ty+th, tx:tx+tw]

	def classify(self):
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		image = clahe.apply(self.image_orig)
		image = cv2.resize(image, (550, 1170), interpolation = cv2.INTER_LANCZOS4)
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		image_preprocessed = (image - 100.090965) / 64.16994
		last_conv, pred = self.model.predict(np.expand_dims(image_preprocessed, axis=0))
		self.pred = pred[0][0]

		last_weight = self.model.layers[-1].get_weights()[0]
		last_conv = last_conv[0]
		predicted_class_weights = last_weight[:, 0]
		heatmap = np.dot(last_conv, predicted_class_weights)
		heatmap = np.maximum(heatmap, 0)
		heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min() + 1e-9) 
		heatmap = cv2.resize(heatmap, (550, 1170))
		heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
		heatmap = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
		cv2.imwrite('heatmap.png', cv2.resize(heatmap, (int(self.image_orig.shape[1]), int(self.image_orig.shape[0]))))
		self.heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

	def toggle_cam(self):
		if(self.heatmap_toggle == 0):
			image = cv2.resize(self.heatmap, (int(self.image_orig.shape[1]//7), int(self.image_orig.shape[0]//7)))
			self.heatmap_toggle = 1
		else:
			image = cv2.resize(self.image_orig, (int(self.image_orig.shape[1]//7), int(self.image_orig.shape[0]//7)))
			self.heatmap_toggle = 0
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		self.img_panel.configure(image=image)
		self.img_panel.image = image

root = tk.Tk()
mammography = mammography_cnn(root)
root.title(mammography.model_name)
root.mainloop()