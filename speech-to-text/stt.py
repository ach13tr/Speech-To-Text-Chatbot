import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

def speech_to_text():
    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Please say something...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Capture the audio

        try:
            # Convert speech to text using Google's speech recognition
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            # Here you can send `text` to your chatbot function
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

if __name__ == "__main__":
    speech_to_text()