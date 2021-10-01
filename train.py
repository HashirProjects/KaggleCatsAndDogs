import os
from tensorflow import keras
import pickle
import random
import numpy as np
#use pickle to import your proccessed data

class ImageClassificationModel():

	def __init__(self, conv, convUnits, filterSize, dense, denseUnits ,inputShape, outputUnits):#add learning rate maybe?

		self.name = f"{conv}c_{convUnits}cu_{filterSize}f_{dense}d_{dense}du"

		if os.path.exists(self.name):

			self.model = keras.models.load_model(self.name)

		else:

			self.model = keras.Sequential()

			self.model.add(keras.layers.Conv2D(convUnits, filterSize, input_shape= inputShape))# one separate conv layer for input
			self.model.add(keras.layers.MaxPooling2D((2,2)))

			for i in range(conv-1):
				self.model.add(keras.layers.Conv2D(convUnits, filterSize))
				self.model.add(keras.layers.MaxPooling2D((2,2)))

			self.model.add(keras.layers.Flatten()) # dense layers can only take flat inputs

			for i in range(dense):
				self.model.add(keras.layers.Dense(denseUnits, activation = "relu"))

			self.model.add(keras.layers.Dense(outputUnits, activation = "softmax"))

			self.model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
			#metrics will always be the same for classification problems so i didnt include them as parameters

	def train(self, X, Y, validationSplit, batchsize, epochs):
		tb = keras.callbacks.TensorBoard(log_dir = f"logs/{self.name}")
		self.model.fit(X, Y, batch_size = batchsize, epochs= epochs, callbacks = [tb], validation_split = validationSplit)
		self.model.save(self.name)

with open("trainingValues.txt", "rb") as file:
	trainingValues = pickle.load(file)

with open("trainingResults.txt", "rb") as file:
	trainingResults = pickle.load(file)

for i in range(1,5):
	for j in range(1,4):
		model = ImageClassificationModel(i, j*16, (3,3),0,0, trainingValues.shape[1:],2)
		model.train(trainingValues,trainingResults,0.1,32,10)


