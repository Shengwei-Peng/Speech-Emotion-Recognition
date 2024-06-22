import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
import utils

warnings.filterwarnings('ignore')
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.compat.v1.InteractiveSession(config=config)
tf.random.set_seed(0)

crema_df = utils.load_crema_dataset('./Crema/')
ravdess_df = utils.load_ravdess_dataset('./Ravdess/audio_speech_actors_01-24')
savee_df = utils.load_savee_dataset('./Savee/')
tess_df = utils.load_tess_dataset('./Tess/')

main_df=pd.concat([crema_df,ravdess_df,savee_df,tess_df],axis=0)
print('All date shape: ', main_df.shape)

plt.figure(figsize=(12,6))
plt.title('Emotions Counts')
emotions=sns.countplot(x='Emotion',data=main_df,palette='Set2', color='red')
emotions.set_xticklabels(emotions.get_xticklabels(),rotation=45)
plt.show()

emotion_names=main_df['Emotion'].unique()

colors={'disgust':'#804E2D','happy':'#F19C0E','sad':'#478FB8','neutral':'#4CB847','fear':'#7D55AA','angry':'#C00808','surprise':'#EE00FF'}
audio_path=[]
for emotion in emotion_names:
    path=np.array(main_df['File_Path'][main_df['Emotion']==emotion])[1]
    data,sr=librosa.load(path)
    utils.wave_plot(data,sr,emotion,colors[emotion])
    utils.spectogram(data,sr,emotion)
    audio_path.append(path)

data,sr=librosa.load(audio_path[0])
plt.figure(figsize=(12,5))
plt.title('Orijinal Audio')
librosa.display.waveshow(data,sr = sr)

noised_audio=utils.add_noise(data)
plt.figure(figsize=(12,5))
plt.title('Noised Audio')
librosa.display.waveshow(noised_audio,sr = sr)
plt.show()

shifted_audio=utils.shifting(data)
plt.figure(figsize=(12,5))
plt.title('Shifted Audio')
librosa.display.waveshow(shifted_audio,sr = sr)
plt.show()

pitched_audio=utils.pitching(data,sr =  sr)
plt.figure(figsize=(12,5))
plt.title('Pitched Audio')
librosa.display.waveshow(pitched_audio,sr = sr)
plt.show()

stretched_audio=utils.streching(data)
plt.figure(figsize=(12,5))
plt.title('Streched Audio')
librosa.display.waveshow(stretched_audio,sr = sr)
plt.show()

stretched_audio=utils.zcr(data,2048,512)
plt.figure(figsize=(12,5))
librosa.display.waveshow(stretched_audio,sr = sr)
plt.show()
stretched_audio=utils.rmse(data,2048,512)
plt.figure(figsize=(12,5))
librosa.display.waveshow(stretched_audio,sr = sr)
plt.show()
stretched_audio=utils.mfcc(data,2048,512)
plt.figure(figsize=(12,5))
librosa.display.waveshow(stretched_audio,sr = sr)
plt.show()

process = False
processed_data_path='./processed_data.csv'
X,Y=[],[]

if process:
    for path,emotion,index in zip(main_df.File_Path,main_df.Emotion,range(main_df.File_Path.shape[0])):
        features=utils.get_features(path)
        print(f'{index} audio has been processed')
        for i in features:
            X.append(i)
            Y.append(emotion)
    print('Done')
    extract=pd.DataFrame(X)
    extract['Emotion']=Y
    extract.to_csv(processed_data_path,index=False)

print('Process data load')
df=pd.read_csv(processed_data_path)
df=df.fillna(0)
X=df.drop(labels='Emotion',axis=1)
Y=df['Emotion']

lb=LabelEncoder()
Y=np_utils.to_categorical(lb.fit_transform(Y))

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=9820020,test_size=0.2,shuffle=True)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=9820020,test_size=0.1,shuffle=True)
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_val=scaler.transform(X_val)

X_train=np.expand_dims(X_train,axis=2)
X_val=np.expand_dims(X_val,axis=2)
X_test=np.expand_dims(X_test,axis=2)

print('Train: {}\nTest: {}\nValidation: {}'.format(X_train.shape, X_test.shape, X_val.shape))


early_stop=EarlyStopping(monitor='val_accuracy',mode='auto',patience=10,restore_best_weights=True)
lr_reduction=ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
best_model=ModelCheckpoint(filepath='best_model.h5',monitor='val_accuracy',mode='max',verbose=1,save_best_only=True)

model=tf.keras.Sequential([
    
    L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    L.Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    
    L.Flatten(),
    L.Dense(512,activation='relu'),
    L.BatchNormalization(),
    L.Dense(256,activation='relu'),
    L.BatchNormalization(),
    L.Dense(128,activation='relu'),
    L.BatchNormalization(),
    L.Dense(7,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
model.summary()

history=model.fit(X_train, y_train, 
                  epochs=100, 
                  validation_data=(X_val,y_val), 
                  batch_size=128,
                  callbacks=[early_stop,lr_reduction,best_model])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['acc','val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Accuracy')
plt.show()

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_check = np.argmax(y_test,axis=1)
loss,accuracy = model.evaluate(X_test,y_test,verbose=1)
print('Test Loss: {}'.format(loss))
print('Test Accuracy: {:.2%}'.format(accuracy))

conf=confusion_matrix(y_check,y_pred)
cm=pd.DataFrame(
    conf,index=[i for i in emotion_names],
    columns=[i for i in emotion_names]
)
plt.figure(figsize=(12,7))
ax=sns.heatmap(cm,annot=True,fmt='d')
ax.set_title('Confusion matrix for model')
plt.show()
print('Model Confusion Matrix\n',classification_report(y_check,y_pred,target_names=emotion_names))

model.save('{:.2%}.h5'.format(accuracy))
