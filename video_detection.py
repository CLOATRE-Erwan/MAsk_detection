import tensorflow as tf
import cv2
import numpy as np
import simpleaudio as sa
from time import time

model = tf.keras.models.load_model(r"C:\Users\utilisateur\Documents\MicrosoftIA\detection_mask\model")

vid = cv2.VideoCapture(0) 
s_img = cv2.imread(r"C:\Users\utilisateur\Documents\MicrosoftIA\detection_mask\resources\sm_happy.png")
s_img=cv2.resize(s_img,(50,50))

v_img = cv2.imread(r"C:\Users\utilisateur\Documents\MicrosoftIA\detection_mask\resources\sm_sad.jpg")
v_img=cv2.resize(v_img,(50,50))

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
wave_obj = sa.WaveObject.from_wave_file("resources/sf_cotillon_01.wav")

timeStart=time()


while(True): 
      
    
    ret, frame = vid.read()
    frame =cv2.addWeighted(frame, 0.8, np.zeros(frame.shape, frame.dtype), 0, 0)
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=7)


    for (x,y,w,h) in faces:

        frame_predict=frame[y-45:y+w+70,x-45:x+h+70]


        frame_predict=cv2.resize(frame_predict,(96,96))

        
        frame_predict=frame_predict.reshape(-1,96,96,3).astype('float')/255
        y_pred=model.predict_classes(frame_predict)



        if y_pred[0]==1:
            
            l_img = frame
            x_offset=y_offset=50
            frame[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
        
        else:

            l_img = frame
            x_offset=y_offset=50
            frame[y_offset:y_offset+v_img.shape[0], x_offset:x_offset+v_img.shape[1]] = v_img

            timeStop=time()
            if timeStop-timeStart >5 :
                timeStart=timeStop
                
                play_obj = wave_obj.play()


    cv2.imshow('frame', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows()