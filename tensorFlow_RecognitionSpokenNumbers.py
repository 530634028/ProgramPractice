#
# Used to recognise spoken numbers with english in chapter 11
# date: 2018-5-23
# a   : zhonghy 
#
#

"""util  tflearn"""
import tflearn
import speech_data
import tensorflow as tf

learning_rate = 0.0001
training_iters = 300000  #iterations number
batch_size = 64

width = 20 #MFCC characteristic
height = 80  #max lenght of voice
classes = 10  #classes of spoken numbers

batch = word_batch = speech_data.mfcc_batch_generator(batch_size) #generate each batch of MFCC
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y

#define LSTM model
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tf.learn.regression(net, optimizer='adam', learning_rate=learning_rate,
                          loss='categorical_crossentropy')

#train model and save it
model = tflearn.DNN(net, tensorboard_verbose=0)
while 1:  #training_iters
    model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY),
              show_metric=True, batch_size=batch_size)
    _y=model.predict(X)
model.save("tflearn.lstm.model")

#predict model
demo_file = "5_Vicki_260.wav"
demo = speech_data.load_wav_file(speech_data.path + demo_file)
result = model.predict([demo])
result = numpy.argmax(result)
print("predicted digit for %s : result = %d "%(demo_file, result))



















