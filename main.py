import sys
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
import io, os
from PIL import Image, ImageDraw
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import numpy as np
import pandas as pd
import pdfplumber
from pdf2image import convert_from_bytes
import uvicorn

app = FastAPI()

# Load the YOLOv5 model
model_path = "./weights/best.pt"
model = attempt_load(model_path)

# Preprocessing function
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def parse_json(pred_json):
    list_dicts = []
    class_map = {0: "answers",1: "difficulty",2: "options",3: "questions",4: "subjects"}
    print("pred_json ", pred_json)
    for i in range(len(pred_json)):
        dict_bboxes = {}
        dict_bboxes['class_id'] = pred_json[i]['class_id']
        dict_bboxes['x-topleft'] = pred_json[i]['bbox'][0]
        dict_bboxes['y-topleft'] = pred_json[i]['bbox'][1]
        dict_bboxes['x-botrght'] = pred_json[i]['bbox'][2]
        dict_bboxes['y-botrght'] = pred_json[i]['bbox'][3]
        dict_bboxes['confidence'] = round(pred_json[i]['confidence'],2)
        list_dicts.append(dict_bboxes)
    df = pd.DataFrame(list_dicts)
    df.sort_values('y-topleft', inplace = True)
    df['class_id'].replace(class_map, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def find_bins(df_options, value_answer):
    options_start = df_options['x-topleft']
    options_end = df_options['x-botrght']
    bin_edges = np.linspace(options_start, options_end, 5)
    slack = 50
    bin_edges[0] =- bin_edges[0] - slack
    bin_edges[4] =+ bin_edges[4] + slack
    options = {0:"a",1:"b",2:"c",3:"d" }
    if value_answer < bin_edges[0] or value_answer > bin_edges[-1]:
        return None

    for i in range(len(bin_edges) - 1):
        if value_answer >= bin_edges[i] and value_answer < bin_edges[i+1]:
            return options[i]
    

    # If the value is equal to the last bin edge, it belongs to the last bin
    return options[len(bin_edges) - 2]

def post_process(df_answers):
    dict_subject = {"a":"yes", "d":"no"}
    dict_difficulty = {"a":"easy", "d":"medium", None:"difficult"}
    df_answers.iloc[-2] = df_answers.iloc[-2].replace(dict_subject)
    df_answers.iloc[-1] = df_answers.iloc[-1].replace(dict_difficulty)
    return df_answers

def update_values(df, response):
    for i in range(1, len(df) + 1):
        q = f"q{i:02d}"
        response[q]['value'] = df.loc[i].option
        response[q]['conf'] = df.loc[i].confidence
    return response

response_template = {
  "userId": 999999,
  "operationId": "abcd-1234-efgh-5678",
  "q01": {"value": "x", "conf": 0},
  "q02": {"value": "x", "conf": 0},
  "q03": {"value": "x", "conf": 0},
  "q04": {"value": "x", "conf": 0},
  "q05": {"value": "x", "conf": 0},
  "q06": {"value": "x", "conf": 0},
  "q07": {"value": "x", "conf": 0},
  "q08": {"value": "x", "conf": 0},
  "q09": {"value": "x", "conf": 0},
  "q10": {"value": "x", "conf": 0},
  "q11": {"value": "xx", "conf": 0},
  "q12": {"value": "xxxxx", "conf": 0}
}

def save_image(image: Image.Image, predictions):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{now}_image.jpg"

    draw = ImageDraw.Draw(image)

    # Get the original image size and the resized image size
    original_size = image.size
    resized_size = (640, 640)

    # Define a color map and a class map
    color_map = {
        "class1": "red",
        "class2": "green",
        "class3": "blue",
        "class4": "yellow",
        "class5": "orange",
    }
    class_map = {
        0: "class1",
        1: "class2",
        2: "class3",
        3: "class4",
        4: "class5",
    }


    # Loop through each prediction and draw the bounding box and class name
    for prediction in predictions:
        bbox = prediction["bbox"]
        class_id = prediction["class_id"]

        # Scale the bounding box coordinates from the resized image to the original image
        bbox[0] = bbox[0] * original_size[0] / resized_size[0]
        bbox[1] = bbox[1] * original_size[1] / resized_size[1]
        bbox[2] = bbox[2] * original_size[0] / resized_size[0]
        bbox[3] = bbox[3] * original_size[1] / resized_size[1]

        # Get the color and description for the class
        color = color_map.get(class_map[class_id], "white")
        description = class_map.get(class_id, "unknown")



        # Draw the bounding box with the color and description
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=3)
        draw.text((bbox[0], bbox[1] - 30), description, fill=color)

    # Save the image with timestamp in the filename
    image.save(filename)
    return filename

def is_pdf(file: UploadFile):
    return file.filename.lower().endswith('.pdf')

def convert_pdf_to_image(file_bytes):
    images = convert_from_bytes(file_bytes)
    if len(images) > 1:
        raise ValueError("The PDF has more than one page.")

    return images[0]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the file bytes
    file_bytes = await file.read()

    # Check if the file is a PDF
    if is_pdf(file):
        try:
            # Convert the PDF to an image
            image = convert_pdf_to_image(file_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Read the image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    
    image_tensor = preprocess_image(image)  # Implement the preprocess_image function as needed

    # Perform inference
    with torch.no_grad():
        pred = model(image_tensor)[0]
        results = non_max_suppression(pred)

    # Extract and format results
    response = []
    for result in results[0].tolist():
        response.append({
            "class_id": int(result[5]),
            "confidence": float(result[4]),
            "bbox": [float(result[0]), float(result[1]), float(result[2]), float(result[3])]
        })
    print("response empty?", response)
    # Save the image with timestamp in the filename and draw the bounding boxes
    filename = save_image(image, response)
    df = parse_json(response)
    df_options = df[df['class_id'] == 'options']
    df_answers = df[df['class_id'] == 'answers']
    if len(df_answers) == 13:
        df_answers['option'] = df_answers['x-topleft'].apply(lambda x: find_bins(df_options, x))
        df_answers = post_process(df_answers)
        df_answers = df_answers[1:]
        df_answers.index = np.arange(1, len(df_answers)+1)
        response = update_values(df_answers, response_template)
    else:
        response = {"error":"incomplete answers"}
    #print(response)

    return JSONResponse(content={"predictions": response})
    #return JSONResponse(content={"filename": filename, "predictions": response})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
