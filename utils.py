import os
import re
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_crema_dataset(path):
    crema=[]
    for wav in os.listdir(path):
        emotion=wav.partition(".wav")[0].split('_')
        if emotion[2]=='SAD':
            crema.append(('sad',path+'/'+wav))
        elif emotion[2]=='ANG':
            crema.append(('angry',path+'/'+wav))
        elif emotion[2]=='DIS':
            crema.append(('disgust',path+'/'+wav))
        elif emotion[2]=='FEA':
            crema.append(('fear',path+'/'+wav))
        elif emotion[2]=='HAP':
            crema.append(('happy',path+'/'+wav))
        elif emotion[2]=='NEU':
            crema.append(('neutral',path+'/'+wav))
        else:
            crema.append(('unknown',path+'/'+wav))
            
    crema_df=pd.DataFrame.from_dict(crema)
    crema_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)
    crema_df['Emotion'].unique()
    return crema_df

def load_ravdess_dataset(path):
    ravdess=[]
    for directory in os.listdir(path):
        actors=os.listdir(os.path.join(path,directory))
        for wav in actors:
            emotion=wav.partition('.wav')[0].split('-')
            emotion_number=int(emotion[2]) 
            ravdess.append((emotion_number,os.path.join(path,directory,wav)))
            
    ravdess_df=pd.DataFrame.from_dict(ravdess)
    ravdess_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)
    ravdess_df['Emotion'].replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'},inplace=True)
    ravdess_df['Emotion'].unique()
    return ravdess_df

def load_savee_dataset(path):
    savee=[]
    for wav in os.listdir(path):
        emo=wav.partition('.wav')[0].split('_')[1].replace(r'[0-9]','')
        emotion=re.split(r'[0-9]',emo)[0]
        if emotion=='a':
            savee.append(('angry',path+'/'+wav))
        elif emotion=='d':
            savee.append(('disgust',path+'/'+wav))
        elif emotion=='f':
            savee.append(('fear',path+'/'+wav))
        elif emotion=='h':
            savee.append(('happy',path+'/'+wav))
        elif emotion=='n':
            savee.append(('neutral',path+'/'+wav))
        elif emotion=='sa':
            savee.append(('sad',path+'/'+wav))
        elif emotion=='su':
            savee.append(('surprise',path+'/'+wav))
            
    savee_df=pd.DataFrame.from_dict(savee)
    savee_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)
    savee_df['Emotion'].unique()
    return savee_df

def load_tess_dataset(path):
    tess=[]

    for directory in os.listdir(path):
        for wav in os.listdir(os.path.join(path,directory)):
            emotion=wav.partition('.wav')[0].split('_')
            if emotion[2]=='ps':
                tess.append(('surprise',os.path.join(path,directory,wav)))
            else:
                tess.append((emotion[2],os.path.join(path,directory,wav)))
                
    tess_df=pd.DataFrame.from_dict(tess)
    tess_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)
    tess_df['Emotion'].unique()
    return tess_df

def wave_plot(data,sr,emotion,color):
    plt.figure(figsize=(12,5))
    plt.title(f'{emotion} emotion for waveplot',size=17)
    librosa.display.waveshow(y=data,sr=sr,color=color)
    plt.show()

def spectogram(data,sr,emotion):
    audio=librosa.stft(data)
    audio_db=librosa.amplitude_to_db(abs(audio))
    plt.figure(figsize=(12,5))
    plt.title(f'{emotion} emotion for spectogram',size=17)
    librosa.display.specshow(audio_db,sr=sr,x_axis='time',y_axis='hz')
    plt.show()

def add_noise(data,random=False,rate=0.035,threshold=0.075):
    if random:
        rate=np.random.random()*threshold
    noise=rate*np.random.uniform()*np.amax(data)
    augmented_data=data+noise*np.random.normal(size=data.shape[0])
    return augmented_data

def shifting(data,rate=1000):
    augmented_data=int(np.random.uniform(low=-5,high=5)*rate)
    augmented_data=np.roll(data,augmented_data)
    return augmented_data

def pitching(data,sr,pitch_factor=0.7,random=False):
    if random:
        pitch_factor=np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(data,sr = sr,n_steps = pitch_factor)

def streching(data,rate=0.8):
    return librosa.effects.time_stretch(data,rate = rate)

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data,sr,flatten:bool=True):
    mfcc=librosa.feature.mfcc(data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_features(data,sr)
    audio=np.array(aud)
    
    noised_audio=add_noise(data,random=True)
    aud2=extract_features(noised_audio,sr)
    audio=np.vstack((audio,aud2))
    
    pitched_audio=pitching(data,sr,random=True)
    aud3=extract_features(pitched_audio,sr)
    audio=np.vstack((audio,aud3))
    
    pitched_audio1=pitching(data,sr,random=True)
    pitched_noised_audio=add_noise(pitched_audio1,random=True)
    aud4=extract_features(pitched_noised_audio,sr)
    audio=np.vstack((audio,aud4))
    
    return audio

def play_audio(audio_path):
    import playsound
    input('Angry Audio Sample')
    playsound(audio_path[5])

    input('Disgust Audio Sample')
    playsound(audio_path[0])

    input('Fear Audio Sample')
    playsound(audio_path[4])

    input('Happy Audio Sample')
    playsound(audio_path[1])

    input('Neutral Audio Sample')
    playsound(audio_path[3])

    input('Sad Audio Sample')
    playsound(audio_path[2])

    input('Surprise Audio Sample')
    playsound(audio_path[6])