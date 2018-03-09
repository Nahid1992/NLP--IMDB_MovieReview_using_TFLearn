import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
#Restarting karnel
import tensorflow as tf
print('Kernel Restarting..')
tf.reset_default_graph()
print('Kernel Restarted..')

input_max_row = 10000
input_max_col = 100
nb_classes = 2

#Dataset Load:
train,valid,test = imdb.load_data(path='imdb.pkl', n_words=input_max_row,valid_portion=0.1)
print('Data Loaded..')
#imdb_review = imdb.load_data(path='imdb.pkl')
#train,test = imdb_review
trainX,trainY = train
testX,testY = test
valX,valY = valid

#data_preprocessing
trainX = pad_sequences(trainX, maxlen=input_max_col)
testX = pad_sequences(testX, maxlen=input_max_col)
valX = pad_sequences(valX, maxlen=input_max_col)

#Binary
trainY = to_categorical(trainY,nb_classes=nb_classes)
testY = to_categorical(testY,nb_classes=nb_classes)
valY = to_categorical(valY,nb_classes=nb_classes)
print('Data is Ready..')


#network
net = tflearn.input_data([None,input_max_col])
net = tflearn.embedding(net,input_dim=input_max_row,output_dim=128)
net = tflearn.lstm(net,128,dropout=0.8)
net = tflearn.fully_connected(net,nb_classes,activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=.001,loss='categorical_crossentropy')
print('Model Created..')

model = tflearn.DNN(net,tensorboard_verbose=3)
model.fit(trainX,trainY,validation_set=(valX,valY),n_epoch=10, show_metric=True,batch_size=64,snapshot_step=200,run_id='IMDBreview_tflearn_run04')

print('Training Completed...')

model.save('models/tflearn_imdb_review.model')
print('Model Saved...')
