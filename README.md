```shell
conda create -y -n deconstruct-mediapipe python=3.9
conda activate deconstruct-mediapipe
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python mediapipe_blendshapes_model_to_pytorch.py
python test_converted_model.py
```
