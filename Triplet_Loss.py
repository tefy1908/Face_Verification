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

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Dot
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras import optimizers
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

PATH = "lfw-deepfunneled/"
USE_SUBSET = True
dirs = sorted(os.listdir(PATH))
if USE_SUBSET:
    dirs = dirs[:500]
    
name_to_classid = {d:i for i,d in enumerate(dirs)}
classid_to_name = {v:k for k,v in name_to_classid.items()}
num_classes = len(name_to_classid)
print("number of classes: "+str(num_classes))

# read all directories
img_paths = {c:[directory + "/" + img for img in sorted(os.listdir(PATH+directory))] 
             for directory,c in name_to_classid.items()}

# retrieve all images
all_images_path = []
for img_list in img_paths.values():
    all_images_path += img_list

# map to integers
path_to_id = {v:k for k,v in enumerate(all_images_path)}
id_to_path = {v:k for k,v in path_to_id.items()}

# build mappings between images and class
classid_to_ids = {k:[path_to_id[path] for path in v] for k,v in img_paths.items()}
id_to_classid = {v:c for c,imgs in classid_to_ids.items() for v in imgs}

from skimage.io import imread
from skimage.transform import resize

def resize100(img):
    return resize(img, (100, 100), preserve_range=True, mode='reflect', anti_aliasing=True)[20:80,20:80,:]

def open_all_images(id_to_path):
    all_imgs = []
    for path in id_to_path.values():
        all_imgs += [np.expand_dims(resize100(imread(PATH+path)),0)]
    return np.vstack(all_imgs)

all_imgs = open_all_images(id_to_path)
mean = np.mean(all_imgs, axis=(0,1,2))
all_imgs -= mean
all_imgs.shape, str(all_imgs.nbytes / 1e6) + "Mo"

def build_pos_pairs_for_id(classid, max_num=50):
    imgs = classid_to_ids[classid]
    if len(imgs) == 1:
        return []
    
    pos_pairs = list(itertools.combinations(imgs, 2))
    
    random.shuffle(pos_pairs)
    return pos_pairs[:max_num]

def build_positive_pairs(class_id_range):
    listX1 = []
    listX2 = []
    
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id)
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
            
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

split_num = int(num_classes * 0.8)

Xa_train, Xp_train = build_positive_pairs(range(0, split_num))
Xa_test, Xp_test = build_positive_pairs(range(split_num, num_classes-1))

# Gather the ids of all images that are used for train and test
all_img_train_idx = list(set(Xa_train) | set(Xp_train))
all_img_test_idx = list(set(Xa_test) | set(Xp_test))

Xa_train.shape, Xp_train.shape

from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
])

class TripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, Xa_train, Xp_train, batch_size, all_imgs, neg_imgs_idx):
        self.cur_img_index = 0
        self.cur_img_pos_index = 0
        self.batch_size = batch_size
        
        self.imgs = all_imgs
        self.Xa = Xa_train  # Anchors
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_imgs_idx = neg_imgs_idx
        
    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):
        low_index = batch_index * self.batch_size
        high_index = (batch_index + 1) * self.batch_size

        imgs_a = self.Xa[low_index:high_index]  # Anchors
        imgs_p = self.Xp[low_index:high_index]  # Positives
        imgs_n = random.sample(self.neg_imgs_idx, imgs_a.shape[0])  # Negatives
            
        imgs_a = seq.augment_images(self.imgs[imgs_a])
        imgs_p = seq.augment_images(self.imgs[imgs_p])
        imgs_n = seq.augment_images(self.imgs[imgs_n])
            
        # We also a null vector as placeholder for output, but it won't be needed:
        return ([imgs_a, imgs_p, imgs_n], np.zeros(shape=(imgs_a.shape[0])))
    
batch_size = 128
gen = TripletGenerator(Xa_train, Xp_train, batch_size, all_imgs, all_img_train_idx)

print(len(all_img_test_idx), len(gen))
[xa, xp, xn], y = gen[0]

print(xa.shape, xp.shape, xn.shape)
plt.figure(figsize=(16, 9))

for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.title("anchor")
    plt.imshow((xa[i] + mean) / 255)
    plt.axis('off')
    
for i in range(5):
    plt.subplot(3, 5, i + 6)
    plt.title("positive")
    plt.imshow((xp[i] + mean) / 255)
    plt.axis('off')
    
for i in range(5):
    plt.subplot(3, 5, i + 11)
    plt.title("negative")
    plt.imshow((xn[i] + mean) / 255)
    plt.axis('off')
    
plt.show()

gen_test = TripletGenerator(Xa_test, Xp_test, 32, all_imgs, all_img_test_idx)
print(len(gen_test))


#triplet Model
# Build a loss which doesn't take into account the y_true, as
# we'll be passing only 0
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

# The real loss is here
def cosine_triplet_loss(X, margin=0.5):
    positive_sim, negative_sim = X
    
    # batch loss
    losses = K.maximum(0.0, negative_sim - positive_sim + margin)
    
    return K.mean(losses)

#sharedeConv
class SharedConv(tf.keras.Model):
    def __init__(self):
        super().__init__(self, name="sharedconv")
        
        self.conv1 = Conv2D(16, 3, activation="relu", padding="same")
        self.conv2 = Conv2D(16, 3, activation="relu", padding="same")
        self.pool1 = MaxPool2D((2,2)) # 30,30
        self.conv3 = Conv2D(32, 3, activation="relu", padding="same")
        self.conv4 = Conv2D(32, 3, activation="relu", padding="same")
        self.pool2 = MaxPool2D((2,2)) # 15,15
        self.conv5 = Conv2D(64, 3, activation="relu", padding="same")
        self.conv6 = Conv2D(64, 3, activation="relu", padding="same")
        self.pool3 = MaxPool2D((2,2)) # 8,8
        self.conv7 = Conv2D(64, 3, activation="relu", padding="same")
        self.conv8 = Conv2D(32, 3, activation="relu", padding="same")
        self.flatten = Flatten()
        self.dropout1 = Dropout(0.2)
        self.fc1 = Dense(40, activation="tanh")
        self.dropout2 = Dropout(0.2)
        self.fc2 = Dense(64)
    
    def call(self, inputs):
        x = self.pool1(self.conv2(self.conv1(inputs)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv6(self.conv5(x)))
        x = self.flatten(self.conv8(self.conv7(x)))
        
        x = self.fc1(self.dropout1(x))
        return self.fc2(self.dropout2(x))
shared_conv = SharedConv()

#TRIPLET NETWORK 
class TripletNetwork(tf.keras.Model):
    def __init__(self, shared_conv):
        super().__init__(self, name="tripletnetwork")
        
        self.shared_conv = shared_conv
        self.dot = Dot(axes=-1, normalize=True)
        self.cosine_triple_loss = Lambda(cosine_triplet_loss, output_shape=(1,))
        
    def call(self, inputs):
        anchor, positive, negative = inputs
        
        anchor = self.shared_conv(anchor)
        positive = self.shared_conv(positive)
        negative = self.shared_conv(negative)

        
        pos_sim = self.dot([anchor, positive])
        neg_sim = self.dot([anchor, negative])

        return self.cosine_triple_loss([pos_sim, neg_sim])
   
model_triplet = TripletNetwork(shared_conv)
model_triplet.compile(loss=identity_loss, optimizer="rmsprop")

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


best_model_fname = "triplet_checkpoint_b2.h5"
best_model_cb = ModelCheckpoint(best_model_fname, monitor='val_loss',
                                save_best_only=True, verbose=1)

history = model_triplet.fit(gen, 
                    epochs=10,
                    validation_data = gen_test,
                    callbacks=[best_model_cb])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 0.5)
plt.legend(loc='best')
plt.title('Loss')
model_triplet.load_weights("triplet_checkpoint_b2.h5")
