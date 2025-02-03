from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import sys
import os

# Ensure src directory is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import inference  # Now it should work

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = inference.predict_cell(file_path)
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
