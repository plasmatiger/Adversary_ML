import keras
import numpy as np

training_data =  #give address
test_data =  #give address

model = Sequential()
#layer 1
#conv+relu
model.add(Convolution2D(32,3,3, input_shape=(64,64,3)))
model.add(Activation('relu'))
#conv+relu
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
#pooling(max)
model.add(MaxPoolin2D(pool_size=(2,2)))
#layer 2
#conv+relu
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
#conv+relu
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
#pooling(max)
model.add(MaxPoolin2D(pool_size=(2,2)))
#layer 3
#conv+relu
model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
#conv+relu
model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
#pooling(max)
model.add(MaxPoolin2D(pool_size=(2,2)))

#preparation for FC(fully_connected)
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#optimizer can also be other. But it is saying, Adam is advanced Gradient Descent

model.compile(loss = 'categorical_crossentropy', optimizer = 'AdamOptimizer', metrics = ['accuracy'])

model.fit_generator(training_data, samples_epoch = 2048, nb_epoch =30, validation_data = test_data, nb_val_samples = 832)

