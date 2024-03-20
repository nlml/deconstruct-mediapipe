# `convert-mediapipe-blendshapes-model-to-pytorch`

This repository contains the code for converting the blendshapes component of MediaPipe's facemesh model to PyTorch.

## Converting the model

Run the following commands:

```shell
conda create -y -n deconstruct-mediapipe python=3.9
conda activate deconstruct-mediapipe
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python mediapipe_blendshapes_model_to_pytorch.py
```

## Checking the converted model

If you want to check the converted model, you need to `pip install mediapipe`. Then run `python test_converted_model.py`.

The outputs should look as follows, showing that the PyTorch model gives the same results for the (first 12) blendshapes as MediaPipe itself.

```shell
--------------------------------- Face 1 --------------------------------
Blendshapes from MediaPipe:
[0.    0.242 0.217 0.001 0.018 0.014 0.    0.    0.    0.098 0.048 0.017]
Blendshapes from PyTorch:
[0.    0.242 0.217 0.001 0.018 0.014 0.    0.    0.    0.098 0.048 0.017]
Blendshapes from TFLite:
[0.    0.242 0.217 0.001 0.018 0.014 0.    0.    0.    0.098 0.048 0.017]
--------------------------------- Face 2 --------------------------------
Blendshapes from MediaPipe:
[0.    0.085 0.169 0.008 0.032 0.013 0.    0.    0.    0.063 0.092 0.216]
Blendshapes from PyTorch:
[0.    0.085 0.169 0.008 0.032 0.013 0.    0.    0.    0.063 0.092 0.216]
Blendshapes from TFLite:
[0.    0.085 0.169 0.008 0.032 0.013 0.    0.    0.    0.063 0.092 0.216]
```
