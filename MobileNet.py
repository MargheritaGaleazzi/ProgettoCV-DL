from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime, os
from pathlib import Path
import numpy as np
import random
import pandas as pd
import glob
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from numba import jit
import PIL

target_shape = (224,224)
embedding_size = 256


base_cnn = tf.keras.applications.MobileNetV3Large(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False)
flatten = tf.keras.layers.Flatten()(base_cnn.output)
dense1 = tf.keras.layers.Dense(embedding_size)(flatten) 
output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) (dense1) 
embeddingNet = tf.keras.Model(base_cnn.input, output, name="Embedding")

def read_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image

data_dir = "trainingSet/"
data_dir = Path(data_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'**/**/*.jpg'), shuffle=False)
image_count = len(list_ds)
label_list = []

for i in (list_ds):
   label_list = label_list + [int(i.numpy().decode().split('\\')[2])]

label_list = tf.data.Dataset.from_tensor_slices(label_list)
dataset = list_ds.map(read_image)
dataset = tf.data.Dataset.zip((dataset, label_list))
dataset = dataset.shuffle(buffer_size=1024)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
  tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
  tf.keras.layers.experimental.preprocessing.RandomRotation(factor = 0.25)
])

logdir = os.path.join("/trainingSet", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    '/nuovo', monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')

def net(optimizer, loss, num_epochs, batch_size, train_dataset,val_dataset,embeddingNet):
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    embeddingNet.compile(
        optimizer=optimizer,
        loss=loss)

    history = embeddingNet.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback,model_checkpoint_callback])

    embeddingNet = tf.keras.models.load_model('/nuovo')

    data_dir = "/gallery"
    data_dir = Path(data_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'**/**/*.jpg'), shuffle=False)
    label_list = []
    for i in (list_ds):
        label_list = label_list + [int(i.numpy().decode().split('/')[-2])]

    label_list = tf.data.Dataset.from_tensor_slices(label_list)
    completeDataset = list_ds.map(read_image)

    completeDataset = completeDataset.batch(batch_size, drop_remainder=False)
    completeDataset = completeDataset.prefetch(8)
    return completeDataset, history, batch_size

#Cambiare i parametri da questa funzione --> net(optimizer, loss, num_epochs, batch_size,train_dataset, val_dataset,embeddingNet)
completeDataset, history, batch_size = net(tf.keras.optimizers.Adam(0.0001),tfa.losses.TripletHardLoss(soft=True),1,64,train_dataset,val_dataset,embeddingNet)

from tensorflow.keras.callbacks import Callback
class PredictionCallback(Callback):
    
    def on_predict_begin(self, logs=None):
        print("Starting prediction ...")
    
    def on_predict_batch_begin(self, batch, logs=None):
        print(f"Prediction: Starting batch {batch}")
        
    def on_predict_batch_end(self, batch, logs=None):
        print(f"Prediction: Finish batch {batch}")
    
    def on_predict_end(self, logs=None):
        print("Finished prediction")

embeddingTotale = embeddingNet.predict(completeDataset, callbacks=[PredictionCallback()])

list_all_images_for_embedding = (list(list_ds.as_numpy_iterator()))

numFotoTotale = len(list_all_images_for_embedding)

EANTotale = []
imageTotale = []
cat_totale = []
for i in list_all_images_for_embedding:
  EANTotale = EANTotale + [i.decode('UTF-8').split('/')[-2]]
  cat_totale = cat_totale + [i.decode('UTF-8').split('/')[-3]]
  imageTotale = imageTotale + [i.decode('UTF-8').split('/')[-1].split('.')[0]]

EAN_imageTotale = []
for i in range(numFotoTotale):
  EAN_imageTotale = EAN_imageTotale + [EANTotale[i] + "_" + imageTotale[i] + "_" + cat_totale[i]]

dictTotale = {key: [] for key in EAN_imageTotale}
for k, v in zip(EAN_imageTotale, embeddingTotale):
    dictTotale[k].append(v)

data_dirTest = "/testSet/"
data_dirTest = Path(data_dirTest)
list_dsTest = tf.data.Dataset.list_files(str(data_dirTest/'**/**/*.jpg'), shuffle=False)
EANListTest = []
imageListTest = []
for i in (list_dsTest):
    EANListTest = EANListTest + [(i.numpy().decode().split('/')[-2])]
    imageListTest = imageListTest + [i.numpy().decode().split('/')[-1]]

datasetTest = list_dsTest.map(read_image)

datasetTest = datasetTest.batch(batch_size, drop_remainder=False)
datasetTest = datasetTest.prefetch(8)

embeddingTest = embeddingNet.predict(datasetTest, callbacks=[PredictionCallback()])

@jit(nopython=True)
def fast_cosine(u, v):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio

matching_EAN = []
matching_image = []
cosine_similarity_array = []
similarity_temp = -1
for i in embeddingTest:
	similarity_temp = -1
	for key, value in dictTotale.items():
		embedding_confronto = (((np.array(value, dtype = np.float32))))
		similarity = fast_cosine(np.array(i,dtype = np.float32), embedding_confronto[0])
		if similarity > similarity_temp:
			similarity_temp = similarity
			moreMatchingKey = key
	print(moreMatchingKey)
	matching_EAN = matching_EAN + [moreMatchingKey.split('_')[0]]
	matching_image = matching_image + [moreMatchingKey.split('_',1)[1] + '.jpg']
	cosine_similarity_array = cosine_similarity_array + [similarity_temp]

def create(lst):
    l = []
    for a, b, c, d in lst:
        l.append((float(a), b, c, d))
    return l

similarity_temp = []
EAN_temp = []
topN = []
topNTotale = []
count = 0
for i in embeddingTest:
	similarity_temp = []
	EAN_temp = []
	imagePath = []
	cat = []
	keyAttuale = list(EANListTest)[count]
	for key, value in dictTotale.items():
		embedding_confronto = (((np.array(value, dtype = np.float32))))
		similarity = fast_cosine(np.array(i), embedding_confronto[0])
		catTemp = key.split('_')[-1]
		PathTemp = '_'.join(key.split('_')[1:-1]) + '.jpg'
		if similarity > 0.2:
				similarity_temp = similarity_temp + [similarity]
				EAN_temp = EAN_temp + [key.split('_')[0]]
				imagePath = imagePath + [PathTemp]
				cat = cat + [catTemp]
	z = np.array(create(zip(similarity_temp,EAN_temp,imagePath,cat)))
	topN = z[np.argsort(z[:, 0].astype('float'))]
	topN = topN[-10:]
	topN = np.flip(topN, axis = 0)
	topN = list(topN)
	topN.append(keyAttuale)
	topN.append(imageListTest[count])
	count = count + 1
	topNTotale = topNTotale + [topN]

totaleCountTop5 = len(topNTotale)
countTop5 = 0
for i in range(totaleCountTop5):
  if (topNTotale[i][0][1]==topNTotale[i][10] or topNTotale[i][1][1]==topNTotale[i][10] or topNTotale[i][2][1]==topNTotale[i][10] or topNTotale[i][3][1]==topNTotale[i][10] or topNTotale[i][4][1]==topNTotale[i][10]):
    countTop5 = countTop5 + 1
print(countTop5/totaleCountTop5)

top10Totale = topNTotale
totaleCountTop10 = len(topNTotale)
countTop10 = 0
for i in range(totaleCountTop10):
  if (topNTotale[i][0][1]==topNTotale[i][10] or topNTotale[i][1][1]==topNTotale[i][10] or topNTotale[i][2][1]==topNTotale[i][10] or topNTotale[i][3][1]==topNTotale[i][10] or topNTotale[i][4][1]==topNTotale[i][10] or topNTotale[i][5][1]==topNTotale[i][10] or topNTotale[i][6][1]==topNTotale[i][10] or topNTotale[i][7][1]==topNTotale[i][10] or topNTotale[i][8][1]==topNTotale[i][10] or topNTotale[i][9][1]==topNTotale[i][10]):
    countTop10 = countTop10 + 1
print(countTop10/totaleCountTop10)

resultCosineTest = pd.DataFrame({'EANTest': EANListTest, 'nameImageTest': imageListTest, 'matchingEAN': matching_EAN, 'nameImageMatch': matching_image, 'cosineSimilarity': cosine_similarity_array})

comparison_column = np.where(resultCosineTest["EANTest"] == resultCosineTest["matchingEAN"], True, False)
resultCosineTest["Same"] = comparison_column
true_count = (resultCosineTest.Same).sum()
accuracy = true_count / len(resultCosineTest)

print(accuracy)

