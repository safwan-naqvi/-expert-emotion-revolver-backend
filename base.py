
from PIL import Image
import glob
import numpy as np

def finalAssumptionVideo(result):
    #Here we will calculate the occurance of Result of (Angry,Happy,Neutral,Sad) and utilize this data to make more classes
    values, counts = np.unique(result, return_counts=True)
    values = values.tolist() #convert to lists
    counts = counts.tolist()
    ind = counts.index(max(counts)) #getting index of max value of count in counts array
    print("First Largest :"+values[ind]) #printing first largest value
    del values[ind] #deleting Element from list of value
    del counts[ind] #deleting Element from list of counts
    ind2 = counts.index(max(counts)) #2nd largest element
    print("Second Largest :"+values[ind2]) #printing second largest value

for Imagename in glob.glob(dir_path,recursive=True):
    print("Image: "+Imagename)
    #faces = RetinaFace.extract_faces(img_path = path+i+".jpg", align = True)
    faces = RetinaFace.extract_faces(img_path = Imagename, align = True)
    embeddings = []
    # Trick for more than one face in video
    for face in faces:
        embedding = DeepFace.represent(img_path = face, model_name = 'Facenet', enforce_detection = False)
        embeddings.append(embedding)
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

print(Video_Results)

finalAssumption(Video_Results)


