from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import re  #re is regix it is used fro data cleaning
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from keras.models import load_model
import pickle
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


"""
1. we are going to make some constraints
A. Make a models path(BiGRU)
B. Make tokenizers' path
C. Max sequence Length (padding wala ki only 50 words aayenge)
D. Emotions labels
E. Emotions emojis
"""
# A. Make a models path(BiGRU)
model_path = 'Artifacts/BiGRU_Model.keras'

# B. Make tokenizers' path
tokenizer_path='Artifacts/tokenizer.pkl'

max_seq_length=50

emotion_labels=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

Emotion_emojis = {
    'sadness':'😢',
    'joy':'🤗',
    'love':'❤️',
    'anger':'😡',
    'fear':'😰',
    'surprise':'😲'
    }

"""
2. Preprocess the upcoming text
A. convert the text to lower Case 
B. remove apausetrophes (ex: can't -> cant)
C. remove special characters and punchuations
D. remove extra spaces

"""

def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]","", text)
    text = re.sub(r"\s+", " ", text).strip() #removes extra spaces
    return text

"""
3. Request and Response Schemas

A. Text Input -> Input Schema the text sent by the user
B. Prediction Response -> Output Schema the emotion to predict
C. Health Response(Server)
"""
class TextInput(BaseModel):
    text : str = Field(..., min_length=1, max_length=2000, description="The sentence to analyse", json_schema_extra={"example": "I feel so happy and excited"} )

class predictionResponse(BaseModel): #output body mein kya kya ana chahiye
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

"""
4. Model Loading and Lifespan Management 
Load the model and tokenizer once the server starts up.
"""  
dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer....")

    dl_model["BiGRU"] = load_model(model_path)

    with open(tokenizer_path, 'rb') as file:
        dl_model["tokenizer"] = pickle.load(file)

    print("Model and tokenizer loaded successfully!")

    yield

    dl_model.clear()

#Lifespan created    

app = FastAPI(lifespan=lifespan)

"""
5. Mount the static file to the Fast API app
A. Enable CORS(Cross-Origin Resource Sharing) to allow the requests from different origin.
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.mount('/static', StaticFiles(directory="static"), name="static")

"""
6. API EndPoints
A. Server at homepage('/')
B. Health Check at Endpoint('/health')
C. Predict Emotion Endpoint('/predict')
"""
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

@app.post('/predict', response_model=predictionResponse)
def predict_emotion(text_input: TextInput):
    #1. Clean the input senteces
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if BiGRU_model is None or tokenizer_model is None: # the raise the http exception with status code 503 please try again later
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later")

    clean_text = preprocess_text(text_input.text)

    #2. Convert the words into the numbers
    #3. Pad the sequences to ensure uniform length
    tokenized_text = tokenizer_model.texts_to_sequences([clean_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen= max_seq_length,
        padding='post',
        truncating='post'
    )
    #4. Run prediciton using BiGRU model
    probabilities = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities)) 
    all_probabilities = {
    label: float(prob)
    for label, prob in zip(emotion_labels, probabilities)
}
    return predictionResponse(text=text_input.text, predicted_emotion=emotion_labels[top_emotion_index],
    confidence = float(probabilities[top_emotion_index]),
    all_probabilities = all_probabilities
    )
    #5. Retum Top emotion and full probability breakdown
    