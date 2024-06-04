import tensorflow as tf

# If you have a GPU, execute the following lines to restrict the amount of VRAM used:
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    print("Using GPU {}".format(gpus[0]))
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
else:
    print("Using CPU")

import os
import random
import itertools

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Dot
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dropout
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#processing the dataset 
PATH = "lfw-deepfunneled/"
USE_SUBSET = True

dirs = sorted(os.listdir(PATH))
if USE_SUBSET:
    dirs = dirs[:500]

name_to_classid = {d: i for i, d in enumerate(dirs)}
classid_to_name = {v: k for k, v in name_to_classid.items()}
num_classes = len(name_to_classid)
 
print("number of classes: ", num_classes)

#mapping all images
# read all directories
img_paths = {c: [PATH + subfolder + "/" + img
                 for img in sorted(os.listdir(PATH + subfolder))] 
             for subfolder, c in name_to_classid.items()}

# retrieve all images
all_images_path = []
for img_list in img_paths.values():
    all_images_path += img_list

# map to integers
path_to_id = {v: k for k, v in enumerate(all_images_path)}
id_to_path = {v: k for k, v in path_to_id.items()}

print(all_images_path[:10])
print(len(all_images_path))

# build mappings between images and class
classid_to_ids = {k: [path_to_id[path] for path in v] for k, v in img_paths.items()}
id_to_classid = {v: c for c, imgs in classid_to_ids.items() for v in imgs}
dict(list(id_to_classid.items())[0:13])


plt.hist([len(v) for k, v in classid_to_ids.items()], bins=range(1, 10))
plt.show()

np.median([len(ids) for ids in classid_to_ids.values()])

[(classid_to_name[x], len(classid_to_ids[x]))
 for x in np.argsort([len(v) for k, v in classid_to_ids.items()])[::-1][:10]]

#siamese Net
#A siamese net takes as input two images  outputs a single value which corresponds to the similarity between x1 and x2


# build pairs of positive image ids for a given classid
def build_pos_pairs_for_id(classid, max_num=50):
    imgs = classid_to_ids[classid]
    
    if len(imgs) == 1:
        return []

    pos_pairs = list(itertools.combinations(imgs, 2))
    
    random.shuffle(pos_pairs)
    return pos_pairs[:max_num]

# build pairs of negative image ids for a given classid
def build_neg_pairs_for_id(classid, classes, max_num=20):
    imgs = classid_to_ids[classid]
    neg_classes_ids = random.sample(classes, max_num+1)
    
    if classid in neg_classes_ids:
        neg_classes_ids.remove(classid)
        
    neg_pairs = []
    for id2 in range(max_num):
        img1 = imgs[random.randint(0, len(imgs) - 1)]
        imgs2 = classid_to_ids[neg_classes_ids[id2]]
        img2 = imgs2[random.randint(0, len(imgs2) - 1)]
        neg_pairs += [(img1, img2)]
        
    return neg_pairs

build_pos_pairs_for_id(5, max_num=10)
build_neg_pairs_for_id(5, list(range(num_classes)), max_num=6)


from skimage.io import imread
from skimage.transform import resize


def resize100(img):
    return resize(
        img, (100, 100), preserve_range=True, mode='reflect', anti_aliasing=True
    )[20:80, 20:80, :]


def open_all_images(id_to_path):
    all_imgs = []
    for path in id_to_path.values():
        all_imgs += [np.expand_dims(resize100(imread(path)), 0)]
    return np.vstack(all_imgs)


all_imgs = open_all_images(id_to_path)
all_imgs.shape
print(f"{all_imgs.nbytes / 1e6} MB")

#The following function builds a large number of positives/negatives pairs (train and test)
def build_train_test_data(split=0.8):
    listX1 = []
    listX2 = []
    listY = []
    split = int(num_classes * split)
    
    # train
    for class_id in range(split):
        pos = build_pos_pairs_for_id(class_id)
        neg = build_neg_pairs_for_id(class_id, list(range(split)))
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
            listY += [1]
        for pair in neg:
            if sum(listY) > len(listY) / 2:
                listX1 += [pair[0]]
                listX2 += [pair[1]]
                listY += [0]
    perm = np.random.permutation(len(listX1))
    X1_ids_train = np.array(listX1)[perm]
    X2_ids_train = np.array(listX2)[perm]
    Y_ids_train = np.array(listY)[perm]
    
    listX1 = []
    listX2 = []
    listY = []
    
    #test
    for id in range(split, num_classes):
        pos = build_pos_pairs_for_id(id)
        neg = build_neg_pairs_for_id(id, list(range(split, num_classes)))
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
            listY += [1]
        for pair in neg:
            if sum(listY) > len(listY) / 2:
                listX1 += [pair[0]]
                listX2 += [pair[1]]
                listY += [0]
    X1_ids_test = np.array(listX1)
    X2_ids_test = np.array(listX2)
    Y_ids_test = np.array(listY)
    return (X1_ids_train, X2_ids_train, Y_ids_train,
            X1_ids_test, X2_ids_test, Y_ids_test)

X1_ids_train, X2_ids_train, train_Y, X1_ids_test, X2_ids_test, test_Y = build_train_test_data()
X1_ids_train.shape, X2_ids_train.shape, train_Y.shape
np.mean(train_Y)
X1_ids_test.shape, X2_ids_test.shape, test_Y.shape
np.mean(test_Y)


#data Augmentation and generator 
from imgaug import augmenters as iaa
import numpy as np

import imgaug.augmenters as iaa

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    #iaa.Flipud(0.2), # vertically flip 20% of all images
    # Improve or worsen the contrast of images.
    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
    # crop some of the images by 0-10% of their height/width
    sometimes(iaa.Crop(percent=(0, 0.1))),
    


    # You can add more transformation like random rotations, random change of luminance, etc.
])

class Generator(tf.keras.utils.Sequence):

    def __init__(self, X1, X2, Y, batch_size, all_imgs):
        self.batch_size = batch_size
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.imgs = all_imgs
        self.num_samples = Y.shape[0]
        
    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):
        """This method returns the `batch_index`-th batch of the dataset.
        
        Keras choose by itself the order in which batches are created, and several may be created
        in the same time using multiprocessing. Therefore, avoid any side-effect in this method!
        """
        low_index = batch_index * self.batch_size
        high_index = (batch_index + 1) * self.batch_size
        
        imgs1 = seq.augment_images(self.imgs[self.X1[low_index:high_index]])
        imgs2 = seq.augment_images(self.imgs[self.X2[low_index:high_index]])
        targets = self.Y[low_index:high_index]
    
        return ([imgs1, imgs2], targets)
    
gen = Generator(X1_ids_train, X2_ids_train, train_Y, 32, all_imgs)
print("Number of batches: {}".format(len(gen)))
[x1, x2], y = gen[0]

print(x1.shape, x2.shape, y.shape)
plt.figure(figsize=(16, 6))

for i in range(6):
    plt.subplot(2, 6, i + 1)
    plt.imshow(x1[i] / 255)
    plt.axis('off')
    
for i in range(6):
    plt.subplot(2, 6, i + 7)
    plt.imshow(x2[i] / 255)
    if y[i]==1.0:
        plt.title("similar")
    else:
        plt.title("different")
    plt.axis('off')
    
plt.show()

# own data augmentations

#test images unaffected by the augmentation 
test_X1 = all_imgs[X1_ids_test]
test_X2 = all_imgs[X2_ids_test]
test_X1.shape, test_X2.shape, test_Y.shape


#Simple convolutional model 
@tf.function
def contrastive_loss(y_true, y_pred, margin=0.25):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    y_true = tf.cast(y_true, "float32")
    return tf.reduce_mean( y_true * tf.square(1 - y_pred) +
                  (1 - y_true) * tf.square(tf.maximum(y_pred - margin, 0)))

@tf.function
def accuracy_sim(y_true, y_pred, threshold=0.5):
    '''Compute classification accuracy with a fixed threshold on similarity.
    '''
    y_thresholded = tf.cast(y_pred > threshold, "float32")
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_thresholded), "float32"))

class SharedConv(tf.keras.Model):
    def __init__(self):
        super(SharedConv, self).__init__(name="sharedconv")
        
        # Define the layers
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.pool1 = MaxPool2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPool2D((2, 2))
        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.pool3 = MaxPool2D((2, 2))
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.dense = Dense(128, activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

shared_conv = SharedConv()

print(all_imgs.shape)
print(shared_conv.predict(all_imgs[:10]).shape)
shared_conv.summary()

class Siamese(tf.keras.Model):
    def __init__(self, shared_conv):
        super().__init__(self, name="siamese")
        self.conv = shared_conv
        self.dot = Dot(axes=-1, normalize=True)

    def call(self, inputs):
        i1, i2 = inputs
        x1 = self.conv(i1)
        x2 = self.conv(i2)
        return self.dot([x1, x2])


model = Siamese(shared_conv)
model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy_sim])


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


best_model_fname = "siamese_checkpoint.h5"
best_model_cb = ModelCheckpoint(best_model_fname, monitor='val_accuracy_sim',
                                save_best_only=True, verbose=1)

model.fit_generator(generator=gen, 
                    epochs=15,
                    validation_data=([test_X1, test_X2], test_Y),
                    callbacks=[best_model_cb], verbose=2)

model.load_weights("siamese_checkpoint.h5")


