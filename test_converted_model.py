import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import numpy as np
import torch

from blendshape_info import BLENDSHAPE_MODEL_LANDMARKS_SUBSET, BLENDSHAPE_NAMES
from mlp_mixer import MediaPipeBlendshapesMLPMixer


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


def init_mpipe_blendshapes_model():
    base_options = python.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task",
        # delegate=mp.tasks.BaseOptions.Delegate.GPU,
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    )
    mp_mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_mode,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=2,
        # result_callback=self.result_callback,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_blendshape_score_by_index(blendshapes, i):
    return [_ for _ in blendshapes if _.index == i][0].score


if __name__ == "__main__":
    # Init MediaPipe model
    mesh_detector = init_mpipe_blendshapes_model()
    # Init TFLite model
    tflite_model = TFLiteModel("face_blendshapes.tflite")
    # Init PyTorch model
    blendshape_model = MediaPipeBlendshapesMLPMixer()
    blendshape_model.load_state_dict(torch.load("face_blendshapes.pth"))

    # Run the image through MediaPipe
    IMAGE_FILE = "image.jpg"
    image_mp = mp.Image.create_from_file(IMAGE_FILE)
    mesh_results = mesh_detector.detect(image_mp)
    # Convert landmarks to numpy
    landmarks_np = []
    for face_idx in range(len(mesh_results.face_landmarks)):
        landmarks_np.append(
            np.array([[i.x, i.y, i.z] for i in mesh_results.face_landmarks[face_idx]])
        )
    landmarks_np = np.array(landmarks_np).astype("float32")
    # Convert blendshapes to numpy
    blendshapes_np = np.array(
        [
            [
                get_blendshape_score_by_index(
                    mesh_results.face_blendshapes[face_idx], i
                )
                for i in range(len(BLENDSHAPE_NAMES))
            ]
            for face_idx in range(len(mesh_results.face_landmarks))
        ]
    )
    img_size = np.array([image_mp.width, image_mp.height])[None, None].astype("float32")
    # Compare the results
    for face_idx in range(len(mesh_results.face_landmarks)):
        print("-" * 33 + f" Face {face_idx + 1} " + "-" * 32)
        print("Blendshapes from MediaPipe:")
        print(blendshapes_np[face_idx].round(3)[:12])
        # Run the image through PyTorch
        lmks_tensor = landmarks_np[
            face_idx : face_idx + 1, BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2
        ]
        scaled_lmks_tensor = lmks_tensor * img_size
        with torch.no_grad():
            pytorch_output = blendshape_model(torch.from_numpy(scaled_lmks_tensor))
        print("Blendshapes from PyTorch:")
        print(pytorch_output.squeeze().detach().numpy().round(3)[:12])
        # Run the image through TFLite
        label = tflite_model.predict(scaled_lmks_tensor)
        print("Blendshapes from TFLite:")
        print(label.round(3)[:12])
