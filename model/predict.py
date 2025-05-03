import os
import tensorflow as tf
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "toxicity_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

charset = [chr(i % 95 + 32) for i in range(398)]
char_to_index = {char: idx for idx, char in enumerate(charset)}
max_length = 72

def predict_toxicity(compound: str) -> str:
    # Preprocess the input accordingly (this is just an example)
    # You likely need tokenization/vectorization here
    input_data = preprocess(compound)
    prediction = model.predict(input_data)
    result = interpret_prediction(prediction)
    return result

def smiles_to_ohe(smiles, charset, max_length):
    one_hot = np.zeros((1, max_length, len(charset)), dtype=np.float32)
    for i, char in enumerate(smiles):
        if i >= max_length:
            break
        idx = char_to_index.get(char, 0)
        one_hot[0, i, idx] = 1.0
    return one_hot

# Dummy stubs for preprocessing
def preprocess(compound: str):
    # Replace this with your actual preprocessing logic
    one_hot_encoded = smiles_to_ohe(compound, charset, max_length)
    ohe = np.array(one_hot_encoded)
    ohe = np.expand_dims(ohe, axis=1)
    return ohe

def interpret_prediction(prediction):
    # Replace this based on your model's output
    return "Toxic" if prediction[0][0] > 0.5 else "Non-Toxic"
