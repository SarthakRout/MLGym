import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt

threshold=75 # Not being used 

def load_data(name):
	f = open(name, 'rb')
	data = f.read()
	magic_number = int((data[0:4]).hex(), 16)
	examples = int((data[4:8]).hex(), 16)
	mat = []
	if magic_number == 2051:
		# Images
		for i in range(examples):
			features = []
			for j in range(28*28):
				pixel = data[i*28*28 + j + 12]
				features.append(pixel/255)
			mat.append(features)
	else:
		# Labels
		for i in range(examples):
			label = data[i+8]
			mat.append(label)
	f.close()
	return mat

def scale(scaler_name, train, test):
	if scaler_name == "StandardScaler":
		scaler = StandardScaler()
	elif scaler_name == "MinMaxScaler":
		scaler = MinMaxScaler()
	scaler.fit(train)
	train_scaled = scaler.transform(train)
	test_scaled = scaler.transform(test)
	return train_scaled , test_scaled

def fit_score(model, xtrain, ytrain, xtest, ytest):
	model.fit(xtrain, ytrain)
	print("Train Accuracy: ", model.score(xtrain, ytrain))
	print("Test Acuuracy: ", model.score(xtest, ytest))

train_data_images = "train-images.idx3-ubyte"
train_data_labels = "train-labels.idx1-ubyte"
test_data_images = "t10k-images.idx3-ubyte"
test_data_labels = "t10k-labels.idx1-ubyte"

X_train = np.array(load_data(train_data_images))
y_train = np.array(load_data(train_data_labels))
X_test = np.array(load_data(test_data_images))
y_test = np.array(load_data(test_data_labels))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Scaling to improve accuracy: MinMaxScaler, StandardScaler
X_train_scaled, X_test_scaled = scale("MinMaxScaler", X_train, X_test)

# Logistic Regression Classifier
classifier = LogisticRegression(random_state=17, max_iter=300)
fit_score(classifier, X_train_scaled, y_train, X_test_scaled, y_test)


# SGD Classifier

classifier = SGDClassifier()
fit_score(classifier, X_train_scaled, y_train, X_test_scaled, y_test)

# To show an image
# plt.imshow(X_train[2].reshape(28, 28), interpolation='nearest')
# plt.show()

