import streamlit as st
import cv2
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class EmotionDetection(VideoTransformerBase):
    def __init__(self):
        self.face_xml = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_xml.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for x, y, w, h in faces:
            detected_face = img[y:y+h, x:x+w]
            emotion = DeepFace.analyze(detected_face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion[0]['dominant_emotion']
            cv2.putText(img, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img

st.title("Real-time Emotion Detection")
st.write("This app uses your webcam to detect emotions in real-time using DeepFace.")

webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetection)
