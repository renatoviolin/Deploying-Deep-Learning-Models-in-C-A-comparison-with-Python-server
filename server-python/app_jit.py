# %% ---------------------------------------------
from PIL import Image
import torchvision
from torchvision import transforms
import torch
from fastapi import FastAPI, Request
from io import BytesIO
import base64
import logging
logger = logging.getLogger('hypercorn.error')


# %% ---------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('../models/script.pt')
model.eval()
model.to(device)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
app = FastAPI()

labels = []
with open('../models/labels.txt', 'r') as file:
    labels = file.read().splitlines()
    file.close()


# %% ---------------------------------------------
def inference(img_b64):
    input_image = Image.open(BytesIO(base64.b64decode(img_b64)))
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch.to(device)).detach().cpu()

    output = torch.softmax(output, 1)
    prob, idx = output.topk(k=1, dim=1)
    result = {}
    result["Prediction"] = labels[idx[0]]
    result["Confidence"] = float(prob.numpy()[0])
    return result


@app.post('/predict')
async def predict(request: Request):
    result = (await request.json())
    request_id = result.get('request_id', '-1')
    img_b64 = result.get('image')
    logger.info(f'Request id: {request_id}')
    out = inference(img_b64)
    return out
