from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Konvolüsyonel Sinir Ağının Başlatılması
classifier = Sequential()

# Step 1 - Convolution (creating Feature maps)
classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = "relu"))

# Step 2 - Max Pooling reduced feature map, maintaining features
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding another convolutional layer to increase accuracy and decrease over fitting
classifier.add(Conv2D(32,(3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening turn the feature map into one vector (column)
classifier.add(Flatten())
 
#Step 4 Full connection
 
classifier.add(Dense(units = 128, activation = "relu" ))
classifier.add(Dense(units =  1, activation = "sigmoid" ))

#compiling convolution neural network

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Bölüm 2 - CNN'nin görüntülere uydurulması.
# ImageDataGenerator'ı çağırın, bu işlevler tüm görüntüleri alır ve değiştirir
# overfitting sorunlarına yardımcı olmak için, target_size'yi daha geniş alana değiştirebilirsiniz
# Ancak bu, bir cpu zamanında bir şekilde sonuç elde etmek için bir gpu'ya ihtiyaç duyacaktır.
# işlemek için muhtemelen günler. Eğer target_size değiştirirseniz, ayrıca ihtiyacınız olacak
# input_shape değerini aynı boyuta getirmek için.
from keras.preprocessing.image ithalat ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/user/Desktop/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'C:/Users/user/Desktop/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
#############################################################################
#Sadece kendi ağınızı eğitmek istiyorsanız koşun
#############################################################################
classifier.fit_generator(
                            training_set,
                            steps_per_epoch=(8000/32),
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=(2000/32))
#############################################################################
#############################################################################

# Heres what I prepared earlier load this before testing data, unless you used 
# the above code and created your own.
from keras.models import load_model
classifier1 = load_model("C:/Users/user/Desktop/model_1.h5")

#save model...really need this after 20/14hrs training!!
# Update to above comment, divided training set and test set to cut processing time drastically
# and still have a good enough result. Code now finishes in about 1 hour :)
classifier.save("C:/Users/user/Desktop/model_1.h5")

# predict against images not in test file or training file
import numpy as np
from keras.preprocessing import image 

# change file to what ever picture you want tested answer will always
# be a cat or dog...so if you try insert an image of a turtle do not be disapointed!
newTest = image.load_img("C:/Users/user/Desktop/single_predictions/cat_or_dog_1.jpg", target_size =(64,64))
newTest = image.img_to_array(newTest)
newTest = np.expand_dims(newTest, axis = 0)
result = classifier.predict_classes(newTest)
training_set.class_indices
if result[0][0] == 1:
    prediction = "Dog"
else:
    prediction = "Cat"

#summary of the network
classifier.summary()
# shows all the weights its trained and used to make predictions
classifier.get_weights()
# reveals what optimizer was used
classifier.optimizer
