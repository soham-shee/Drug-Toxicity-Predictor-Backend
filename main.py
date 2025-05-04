from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.predict import predict_toxicity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompoundRequest(BaseModel):
    compound: str

@app.post("/predict/")
def predict_compound(request: CompoundRequest):
    result = predict_toxicity(request.compound)
    return {"prediction": result}

 import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # default to 8000 if not set
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


