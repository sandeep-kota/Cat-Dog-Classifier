import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def main():
	# create generator
	train_datagen = ImageDataGenerator(
		validation_split = 0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		rotation_range=90,
		horizontal_flip=True,
		data_format='channels_last')	# prepare an iterators for each dataset

	test_datagen = ImageDataGenerator(rescale=1./255)


	train_generator = train_datagen.flow_from_directory(
	'../Data/train/',
	target_size=(224,224),
	batch_size=64,
	class_mode='categorical')

	test_generator = test_datagen.flow_from_directory(
	'../Data/train/',
	target_size=(224,224))

	# batchX, batchY = train_generator.next()
	# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
	# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchY.shape, batchY.min(), batchY.max()))

	model = Sequential()
	model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=2, activation="softmax"))

	opt = Adam(lr=0.001)
	model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

	model.summary()

	log_dir = "./Logs/"


	checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,update_freq='epoch')
	hist = model.fit_generator(steps_per_epoch=64,generator=train_generator, validation_data= test_generator, validation_steps=10,epochs=30,callbacks=[checkpoint,early,tensorboard_callback])
        model.save('vgg_Net1.h5')

if __name__=='__main__':
	main()




