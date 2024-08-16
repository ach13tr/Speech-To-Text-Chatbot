
import nltk
import numpy as np
import json
import random
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer

# Download NLTK data if necessary
nltk.download("punkt")

# Load intents.json
with open('files/intents.json') as intents:
    data = json.load(intents)

stemmer = LancasterStemmer()

# Initialize data structures
words = []
labels = []
x_docs = []
y_docs = []

# Process each intent pattern
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        x_docs.append(wrds)
        y_docs.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Stem and lower the words, remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Prepare training data
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(x_docs):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        bag.append(1 if w in wrds else 0)

    output_row = out_empty[:]
    output_row[labels.index(y_docs[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Define the Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(len(training[0]),), activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=500, batch_size=8)

# Save the trained model
model.save('model.h5')

# Function to convert user input into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:
        for i, w in enumerate(words):
            if w == s_word:
                bag[i] = 1

    return np.array(bag)

# Chat function
def chat():
    print("Start chatting with the bot (type 'quit' to stop)!")
    
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break

        # Predict the response using the trained model
        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Find the response based on the predicted tag
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
                print("Bot:\t" + random.choice(responses))

# Run the chat function
chat()
