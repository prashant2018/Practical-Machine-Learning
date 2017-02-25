from __future__ import print_function,division,absolute_import
import logging
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from tensorflow.examples.tutorials.mnist import input_data

logging.getLogger().setLevel(logging.INFO)

#Step 1: Load Data
mnist = input_data.read_data_sets("/temp/data")

steps = 2000

train_X = mnist.train.images[:54000]
test_X = mnist.test.images[:9000]

train_y = mnist.train.labels[:54000]
test_y = mnist.test.labels[:9000]

train_y = np.asarray(train_y,dtype=np.int32)
test_y = np.asarray(test_y,dtype=np.int32)


#Step 2: Build DNN Classifier
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]

#Build 3 layer DNN with 10, 20, 10 units respectively
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
											hidden_units=[10,20,10],
											n_classes=10,
											model_dir="/temp/mnist_model4")
#Fit Model
classifier.fit(x=train_X,y=train_y,steps=steps)

#Step 4: Accuracy
accuracy_score = classifier.evaluate(x=test_X,y=test_y)['accuracy']

print('Accuracy : {0:f}'.format(accuracy_score))

# Steps 200  , Acc : 87.3%
# Steps 2000 , Acc - 93.7 %