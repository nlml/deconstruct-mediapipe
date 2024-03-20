"""
Re-implementation of the MLP-Mixer blendshapes prediction model from
MediaPipe. See README.md for more details.

The MediaPipeBlendshapesMLPMixer class' forward method has an expected
input shape of: (batch_size, 146, 2).
146 here refers to this subset of face mesh landmarks output by MediaPipe:
0,   1,   4,   5,   6,   7,   8,   10,  13,  14,  17,  21,  33,  37,  39,
40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
81,  82,  84,  87,  88,  91,  93,  95,  103, 105, 107, 109, 127, 132, 133,
136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477

The 2 represents the x and y coordinates of the landmarks.
The normalized landmarks that are output by mediapipe need to be multiplied
by the image height and image width before being passed to this model.
The scale doesn't matter, as it is normalized out by this model, but the
aspect ratio of the x to y landmark coordinates needs to be correct.

Example usage:
```
blendshape_model = MediaPipeBlendshapesMLPMixer()
mesh_detector = init_mpipe_blendshapes_model()
mesh_results = mesh_detector.detect(image_mp)
landmarks_np = []
for face_idx in range(len(mesh_results.face_landmarks)):
    landmarks_np.append(np.array([[i.x, i.y, i.z] for i in mesh_results.face_landmarks[face_idx]]))
landmarks_np = np.array(landmarks_np).astype('float32')
lmks_tensor = landmarks_np[:1, BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2]
img_size = np.array([image_mp.width, image_mp.height])[None, None].astype('float32')
scaled_lmks_tensor = torch.from_numpy(lmks_tensor * img_size)
output = blendshape_model(scaled_lmks_tensor)
```
"""

import torch
import torch.nn as nn


class MLPMixerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        num_patches,
        hidden_units_mlp1,
        hidden_units_mlp2,
        dropout_rate=0.0,
        eps1=0.0000010132789611816406,
        eps2=0.0000010132789611816406,
    ):
        super().__init__()
        self.mlp_token_mixing = nn.Sequential(
            nn.Conv2d(num_patches, hidden_units_mlp1, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_units_mlp1, num_patches, 1),
        )
        self.mlp_channel_mixing = nn.Sequential(
            nn.Conv2d(in_dim, hidden_units_mlp2, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_units_mlp2, in_dim, 1),
        )
        self.norm1 = nn.LayerNorm(in_dim, bias=False, elementwise_affine=True, eps=eps1)
        self.norm2 = nn.LayerNorm(in_dim, bias=False, elementwise_affine=True, eps=eps2)

    def forward(self, x):
        x_1 = self.norm1(x)
        mlp1_outputs = self.mlp_token_mixing(x_1)
        x = x + mlp1_outputs
        x_2 = self.norm2(x)
        mlp2_outputs = self.mlp_channel_mixing(x_2.permute(0, 3, 2, 1))
        x = x + mlp2_outputs.permute(0, 3, 2, 1)
        return x


class MediaPipeBlendshapesMLPMixer(nn.Module):
    def __init__(
        self,
        in_dim=64,
        num_patches=97,
        hidden_units_mlp1=384,
        hidden_units_mlp2=256,
        num_blocks=4,
        dropout_rate=0.0,
        output_dim=52,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(146, 96, kernel_size=1)
        self.conv2 = nn.Conv2d(2, 64, kernel_size=1)
        self.extra_token = nn.Parameter(torch.randn(1, 64, 1, 1), requires_grad=True)
        self.mlpmixer_blocks = nn.Sequential(
            *[
                MLPMixerLayer(
                    in_dim,
                    num_patches,
                    hidden_units_mlp1,
                    hidden_units_mlp2,
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mlp = nn.Conv2d(in_dim, output_dim, 1)

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / x.norm(dim=2, keepdim=True).mean(1, keepdim=True)
        x = x.unsqueeze(-2) * 0.5
        x = self.conv1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv2(x)
        x = torch.cat([self.extra_token, x], dim=3)
        x = x.permute(0, 3, 2, 1)
        x = self.mlpmixer_blocks(x)
        x = x.permute(0, 3, 2, 1)
        x = x[:, :, :, :1]
        x = self.output_mlp(x)
        x = torch.sigmoid(x)
        return x#.squeeze()


if __name__ == "__main__":
    model = MediaPipeBlendshapesMLPMixer()
    print(model)
    input_tensor = torch.randn(1, 146, 2)
    output = model(input_tensor)
    print(output.shape)
