import threading
import time
import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import atexit
from transformers import pipeline, Conversation

# Load model
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

# Load pretrained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chatbot = pipeline("conversational", model=model, tokenizer=tokenizer)

# Initialize chat history
chat_history_file = "chat_history.txt"


def load_chat_history():
    try:
        with open(chat_history_file, "r") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return []


def save_chat_history(chat_history):
    with open(chat_history_file, "w") as file:
        for line in chat_history:
            file.write(line + "\n")


def clear_chat_history():
    if os.path.exists(chat_history_file):
        os.remove(chat_history_file)


# Create chat history file if not exists
if not os.path.exists(chat_history_file):
    open(chat_history_file, "w").close()

chat_history = load_chat_history()

emotion_list = []
emotion_thread_started = False


def start_emotion_thread():
    global emotion_thread_started
    if not emotion_thread_started:
        emotion_thread = threading.Thread(target=process_emotions)
        emotion_thread.start()
        emotion_thread_started = True


def process_emotions():
    fl = True
    while fl:
        time.sleep(10)
        if len(emotion_list) > 0:
            # Find the most common emotion
            most_common_emotion = max(set(emotion_list), key=emotion_list.count)

            # Save a message based on the most common emotion
            if most_common_emotion == "sad":
                response = "You seem to be feeling sad. How can I help you?"
                chat_history.append(f"Chatbot: {response}")
                save_chat_history(chat_history)
                # st.session_state.chatbot_response = response
            else:
                response = f"You seem to be feeling {most_common_emotion}. How can I help you?"
                chat_history.append(f"Chatbot: {response}")
                save_chat_history(chat_history)
            fl = False

        # Clear the emotion list
        emotion_list.clear()


def chatbot_response(user_input):
    chat_history.append(f"User: {user_input}")
    if 'conversion' not in st.session_state.keys():
        conversation_user = Conversation(str(user_input))
        conversation = chatbot(conversation_user)
        bot_response = conversation.generated_responses[-1]
        chat_history.append(f"Chatbot: {bot_response}")
        save_chat_history(chat_history)
        st.session_state['conversion'] = conversation
    else:
        conversation = st.session_state['conversion']
        conversation.add_user_input(str(user_input))
        conversation = chatbot(conversation)
        bot_response = conversation.generated_responses[-1]
        chat_history.append(f"Chatbot: {bot_response}")
        save_chat_history(chat_history)
        st.session_state['conversion'] = conversation
    return bot_response

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        # print('from frame')
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            emotion_list.append(output)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        start_emotion_thread()
        return img


def main():
    st.session_state['emotion_flag'] = True
    st.title("Real Time Face Emotion Detection with Chatbot Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown("Feel free to chat with the chatbot!")
    text = st.sidebar.empty()
    user_input = text.text_input("You:", key="user_input")

    if user_input:
        bot_response = chatbot_response(user_input)


    # Display chat history
    st.sidebar.markdown("---")
    for idx, message in enumerate(chat_history):
        role, text = message.split(": ", 1)
        if role == "User":
            st.sidebar.text_input("You:", value=text, key=f"user_input_{idx}", disabled=True)
        else:
            st.sidebar.text_area("Chatbot:", value=text, key=f"bot_response_{idx}", disabled=True, height=50)

    if choice == "Home":
        st.write("This application has two functionalities:")
        st.write("1. Real-time face detection using webcam feed.")
        st.write("2. Real-time face emotion recognition.")
        st.write("3. Real-time chat with chatBot.")

    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("About this app")
        st.markdown(
            "This application is developed by Sabbir Hosen using Streamlit Framework, OpenCV, "
            "Transformer, HuggingFace, Custom CNN model, Microsoft Dilogpt  and Keras library for demonstration purposes.")
        st.markdown("If you have any suggestions or want to comment, please write an email to "
                    "sabbircse44@gmail.com")
    else:
        pass


if __name__ == "__main__":
    atexit.register(clear_chat_history)
    main()
