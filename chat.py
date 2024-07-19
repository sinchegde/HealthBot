import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize model and load trained state
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "HealthBot"

def get_response(msg):
    sentence = tokenize(msg)  # Tokenize user input
    X = bag_of_words(sentence, all_words)  # Convert to bag of words
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)  # Convert to tensor

    output = model(X)  # Get model prediction
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]  # Get predicted tag

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]  # Get probability of predicted tag
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])  # Return random response for the tag

    return "I do not understand..."  # Default response if confidence is low

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
