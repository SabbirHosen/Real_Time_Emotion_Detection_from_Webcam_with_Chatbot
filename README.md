# Real-Time Emotion Detection and Chatbot using Streamlit

This Streamlit project demonstrates real-time emotion detection from a webcam feed and interactive chat with a chatbot using a custom CNN model for emotion detection and Hugging Face's transformer model for conversation using Microsoft DialoGPT.

## Features

- Real-time emotion detection from webcam feed.
- Chat with an interactive chatbot powered by Microsoft DialoGPT.
- Custom CNN model for emotion detection (angry, happy, neutral, sad, surprise).
- Streamlit's user-friendly interface for easy interaction.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SabbirHosen/Real_Time_Emotion_Detection_from_Webcam_with_Chatbot.git
   cd Real_Time_Emotion_Detection_from_Webcam_with_Chatbot
   ```

2. Create and activate the virtual environment:
   ```bash
   python3 -m venv venv_name
   source venv_name/bin/activate
   ```
   Replace 'venv_name' with the desired name for your virtual environment.

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   At first time it will take some time to load the project. Because it will download all the necessary models and files.

## Usage

1. Upon running the Streamlit app, you will be presented with a sidebar containing different options:
   - **Home**: Overview of the application.
   - **Webcam Face Detection**: Starts the webcam feed for real-time emotion detection.
   - **Chat with Chatbot**: Allows you to chat with the chatbot using Microsoft DialoGPT.

2. In the "Webcam Face Detection" section, click the "Start" button to begin real-time emotion detection. Your facial expressions will be analyzed to detect emotions (angry, happy, neutral, sad, surprise).

3. In the "Chat with Chatbot" section, you can interact with the chatbot using the text input box. The chatbot will respond based on the conversation history and context provided by Microsoft DialoGPT.

## Technologies Used

- Streamlit: A Python library for creating interactive web applications for data science.
- OpenCV: Used for webcam feed and image processing.
- Hugging Face Transformers: Utilized for integrating the Microsoft DialoGPT model for conversation.
- Custom CNN Model: Developed for real-time emotion detection from facial expressions.

## Acknowledgments

- Emotion detection CNN model inspired by various open-source emotion detection projects.
- Hugging Face for providing access to pre-trained transformer models.
- Streamlit for creating a user-friendly and interactive interface.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to contribute, modify, or enhance this project. If you have any suggestions or feedback, please create an issue or contact the project owner at sabbircse44@gmail.com.