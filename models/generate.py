# %% ---------------------------------------------
import torch
import torchvision


# %% ---------------------------------------------
model = torchvision.models.resnet152(pretrained=True)
model.eval()
for param in model.parameters():
    param.requires_grad = False


# %% ---------------------------------------------
with torch.no_grad():
    inp = torch.randn(1, 3, 224, 224)
    traced_script = torch.jit.trace(model, inp)
    traced_script.save('trace.pt')


# %% ---------------------------------------------
with torch.no_grad():
    traced_script = torch.jit.script(model)
    traced_script.save('script.pt')


# %% ---------------------------------------------
import torch.onnx
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224)
torch_out = model(x)
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "onnx.pt",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

# %%
