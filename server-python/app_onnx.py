# %% ---------------------------------------------
from fastapi import FastAPI, Request
from io import BytesIO
import base64
from PIL import Image
import onnxruntime
from torchvision import transforms
import numpy as np
import logging
logger = logging.getLogger('hypercorn.error')


# %% ---------------------------------------------
app = FastAPI()
onnx_session = onnxruntime.InferenceSession("../models/model.onnx")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


labels = []
with open('../models/labels.txt', 'r') as file:
    labels = file.read().splitlines()
    file.close()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# %% ---------------------------------------------
def inference(img_b64):
    input_image = Image.open(BytesIO(base64.b64decode(img_b64)))
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    ort_inputs = {onnx_session.get_inputs()[0].name: input_batch.numpy()}
    ort_outs = onnx_session.run(None, ort_inputs)
    output = np.array(ort_outs)[0][0]
    output = softmax(output)
    idx = output.argmax()
    prob = output[idx]
    result = {}
    result["Prediction"] = labels[idx]
    result["Confidence"] = float(prob)
    return result


@app.post('/predict')
async def predict(request: Request):
    result = (await request.json())
    request_id = result.get('request_id', '-1')
    img_b64 = result.get('image')
    logger.info(f'Request id: {request_id}')
    out = inference(img_b64)
    return out
