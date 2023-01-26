import pandas as pd
import numpy as np
##########################################
import os
import glob
import tensorflow.python.keras as keras
from keras.models import load_model
import tensorflow as tf
import librosa
import librosa.display
from keras import models, layers, optimizers
from pydub import AudioSegment 
from pydub.utils import make_chunks
from collections import Counter
############################################
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
from flask import Flask
from flask import request
import json
import urllib.request
import requests
import uuid
import os
from deepface import DeepFace
from retinaface import RetinaFace
import cv2
from flask import jsonify
################## FLASK ###################
app = Flask(__name__)
############## Convert Extension ###################
def ConvertToWav(AUDIO_PATH):
    m4a_file = AUDIO_PATH
    wav_filename = "audioConverted.wav"
    track = AudioSegment.from_file(m4a_file,  format= 'm4a')
    file_handle = track.export(wav_filename, format='wav')
    chunks = MakeChunks(file_handle)
    return chunks

################## Make Audio Chunks ###################
def MakeChunks(wav_filename):
    #region Deleting All Files before making new chunks
    dir = 'chunks/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    #endregion

    #region Making Chunks From wavFile That we have
    src = wav_filename
    
    myaudio = AudioSegment.from_file(src , "wav") 
    chunk_length_ms = 1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #endregion

    #region Export all of the individual chunks as wav files in Chunk folder

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print ("exporting", chunk_name)
        chunk.export('chunks/'+chunk_name, format="wav")

    #endregion

    #region Now Calling Prediction on all those files
    path = 'chunks/'
    preds = Prediction(path)
    print(preds)
    #endregion
    finalAssume = finalAssumption(preds)
    #region Passing Predicted values to gather more info about user
    return finalAssume
    #endregion

##################################################
############### Save Audio File into folder to chunks
############### And every chunk will pe passed to librosa
##############################################
############### Save File into PC before converting into frames
############### So every frame will be passed from Deep Face
##############################################
def save_file(uploaded_file):
    try:
        #url_response = urllib.request.urlopen(uploaded_file)
        var = requests.get(uploaded_file)
        filename = str(uuid.uuid4())+".m4a"
        path = os.path.join('uploaded_audio',filename)
        with open(path,'wb') as f:
            f.write(var.content)
        return path
    except Exception as e:
        print(e)
        return 0

def save_video_file(uploaded_file):
    try:
        #url_response = urllib.request.urlopen(uploaded_file)
        var = requests.get(uploaded_file)
        filename = str(uuid.uuid4())+".mp4"
        path = os.path.join('uploaded_video',filename)
        with open(path,'wb') as f:
            f.write(var.content)
        return path
    except Exception as e:
        print(e)
        return 0


################# Predict Emotion from Audio Result ############################
def Prediction(pathAudio):
    Results = []
    files = librosa.util.find_files(pathAudio, ext=['wav']) 
    files = np.asarray(files)
    
    for y in files:
        signal, sample_rate = librosa.load(y, sr=22050, duration=1.00)
        #y, sr = librosa.load(y, sr=22050, duration=1.00)
        if(librosa.get_duration(y=signal, sr=sample_rate)==1.0):
            mfcc = librosa.feature.mfcc(signal,
                                        sample_rate, 
                                        n_mfcc=13, 
                                        n_fft=2048, 
                                        hop_length=512)

            mfcc = mfcc.T
            #mfcc.shape[0]
            mfcc = mfcc.reshape(1, 44, mfcc.shape[1])
            
            model = load_model("Mix_model.h5")
            
            result = model.predict(mfcc)
            class_result = np.argmax(result)
            
            if (class_result == 0):
                Results.append('Angry')
            elif (class_result == 1):
                Results.append('Happy')
            elif (class_result == 2):
                Results.append('Neutral')
            elif (class_result == 3):
                Results.append('Sad')

    return Results

################## Final Audio Assumption #####################

def finalAssumption(result):

    #Here we will calculate the occurance of Result of (Angry,Happy,Neutral,Sad) and utilize this data to make more classes
    values, counts = np.unique(result, return_counts=True)
    values = values.tolist() #convert to lists
    counts = counts.tolist()
    ind = counts.index(max(counts)) #getting index of max value of count in counts array
    # print(values[ind]) #printing first largest value
    # del values[ind] #deleting Element from list of value
    # del counts[ind] #deleting Element from list of counts
    # ind2 = counts.index(max(counts)) #2nd largest element
    # print(values[ind2]) #printing second largest value
    assume = values[ind]

    return assume
    #Here you can make other emotions according to face recognition as well

################### Flask Server Routes ###################

def convertToFrames(video_path):

    vid = cv2.VideoCapture(video_path)
    i = 0
    # a variable to set how many frames you want to skip
    frame_skip = 30
    # a variable to keep track of the frame to be saved
    frame_count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            cv2.imwrite('./data/frame'+str(frame_count*frame_skip)+'.jpg', frame)
            i = 0
            continue
        i += 1

    vid.release()
    cv2.destroyAllWindows()

    Video_Results = [] #Variable for Emotions Result
    dir_path = r'C:\Users\safwa\Desktop\Socion-Datasets\data\*.jpg*'

    for Imagename in glob.glob(dir_path,recursive=True):
        print("Image: "+Imagename)
        faces = RetinaFace.extract_faces(img_path = Imagename, align = True)
    
        # Trick for more than one face in video
        for face in faces:
            # embedding = DeepFace.represent(img_path = face, model_name = 'Facenet', enforce_detection = False)
            # embeddings.append(embedding)
            obj = DeepFace.analyze(img_path = face, actions = ['emotion'], enforce_detection=False)
            angry_emote = {"angry","disgust","fear"}
            happy_emote = {"happy","surprise"}
            if obj['dominant_emotion'] in angry_emote:
                Video_Results.append('Angry')
            elif obj['dominant_emotion'] in happy_emote:
                Video_Results.append('Happy')
            elif obj['dominant_emotion'] == "sad":
                Video_Results.append('Sad')
            else:
                Video_Results.append('Neutral')

    return finalAssumptionVideo(Video_Results)


def finalAssumptionVideo(result):
    #Here we will calculate the occurance of Result of (Angry,Happy,Neutral,Sad) and utilize this data to make more classes
    values, counts = np.unique(result, return_counts=True)
    values = values.tolist() #convert to lists
    counts = counts.tolist()
    ind = counts.index(max(counts)) #getting index of max value of count in counts array
    print("First Largest :"+values[ind]) #printing first largest value
    # del values[ind] #deleting Element from list of value
    # del counts[ind] #deleting Element from list of counts
    # ind2 = counts.index(max(counts)) #2nd largest element
    # print("Second Largest :"+values[ind2]) #printing second largest value
    
    return values[ind]


@app.route('/')
def welcome():
    print(__name__)
    return "Hello! Emotion Detection System"

@app.route('/convert',methods=["POST"])
def emotion():
    
    url = request.form['url'] #URL of Audio
    # print(url)
    filename = save_file(url)
    emotion = ConvertToWav(filename)
    print(emotion)
    return jsonify(
        emote=emotion
    )
    
    

@app.route('/convertVideo',methods=["POST"])
def emotionVideo():
    video_url = request.form['video'] #URL of Video
    print(video_url)
    video_name = save_video_file(video_url)
    emotion = convertToFrames(video_name)
    print(emotion)
    return emotion


#Loading Model of Embeeding and Filenames
feature_list = pickle.load(open('features.pkl','rb'))
filenames = pickle.load(open('movies.pkl','rb'))


def recommend_collaborative(movie_features_df,movies_df,movie_name):
    recommendedItem = {
        "recommend":
        [
            
        ]
    }
    
    movie_features_df_matrix = csr_matrix(movie_features_df.values) #features 
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(movie_features_df_matrix)
    
    query_index = np.where(movie_features_df.index==movie_name)[0][0]
    distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 7)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
        else:
            #print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))
            #print('{0}: {1}, with distance of {2}:'.format(i, movies_df.index[indices.flatten()[i]], distances.flatten()[i])) 
            data = movies_df.iloc[indices.flatten()[i]]
            #data = df.iloc[val[i]]
            data = list(data)
            item_id = str(data[0])
            Title = str(data[1])
            Director = str(data[2])
            Stars = str(data[3])
            IMDB = str(data[4])
            Category = str(data[5])
            Duration = str(data[6])
            Censor = str(data[7])
            Released = str(data[8])
            videoID = str(data[9])
            thumbnail = str(data[10])
            emotion = str(data[11])
            itemRecommendation = {
                "item_id":item_id,
                "title":Title,
                "director":Director,
                "stars":Stars,
                "imdb":IMDB,
                "category":Category,
                "duration":Duration,
                "censor":Censor,
                "released":Released,
                "videoid":videoID,
                "thumbnail":thumbnail,
                "emotion":emotion
            }
            recommendedItem["recommend"].append(itemRecommendation)

     
    return recommendedItem


@app.route('/collaborative',methods=["POST"])
def youtube():
    movie_name = request.form['url'] #URL of Audio
    # print(url)
    movies = recommend_collaborative(feature_list,filenames,movie_name)
    json_object = json.dumps(movies)
    return json_object


if __name__ == "__main__":
    app.run(host='0.0.0.0')


