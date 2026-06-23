import numpy as np

def load_dataset(prefix=''):
	allSubjs = np.arange(0, 60)
	SubjLables = np.append(np.ones((30,),int), np.zeros((30,),int))

	DM = randint(0, 14) #Depressed Males
	DF = randint(15, 29) #Depressed Females
	CM = randint(30, 44) #Control Males
	CF = randint(45, 59) #Control Females
	# testSubjs = [14, 29, 44, 59] #72%
	# testSubjs = [13, 28, 43, 58] #31%
	# testSubjs = [12, 27, 42, 57] # 92% epoch 4, 80% epoch 5
	testSubjs = [DM, DF, CM, CF]
	print('selected test subjects:')
	print(testSubjs)
	testSubjsLables=SubjLables[testSubjs]

	trainSubjs = np.delete(allSubjs, testSubjs)
	trainSubjsLables=SubjLables[trainSubjs]

	# load all train
	trainX, trainy = timesteps_dataset(trainSubjs, trainSubjsLables)
	trainX, trainy = shuffle(trainX, trainy)
	print(trainX.shape, trainy.shape)
	sumary_data(trainy)

	trainX, trainy = resample_data(trainX, trainy)
	print(trainX.shape, trainy.shape)
	train_class_weight = sumary_data(trainy)

	# load all test
	testX, testy = timesteps_dataset(testSubjs, testSubjsLables)
	print(testX.shape, testy.shape)
	test_class_weight = sumary_data(testy)

	testX, testy = resample_data(testX, testy)
	print(testX.shape, testy.shape)
	test_class_weight = sumary_data(testy)

	# one hot encode y
	# trainy = tf.keras.utils.to_categorical(trainy)
	# testy = tf.keras.utils.to_categorical(testy)
	return trainX, trainy, testX, testy, train_class_weight, test_class_weight
