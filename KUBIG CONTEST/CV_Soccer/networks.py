import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
# from sklearn.metrics import average_precision_score, confusion_matrix
# from torcheval.metrics.functional import mean_average_precision
from loupe import NetRVLAD, NetVLAD, PoolingBaseModel
from fusion import CrossModalGatedAttention, CrossModalGatedBottleneckAttention, GatedBimodalTorch, LateFusionTransformer

# ---

## VideoNetwork (PyTorch)

class VideoNetwork(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(VideoNetwork, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        if "RVLAD" in network_type.upper():
            self.vlad_layer = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
        elif "VLAD" == network_type.upper():
            self.vlad_layer = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input):
        x = video_input
        batch_size, num_frames, feature_size = x.shape
        x = x.view(-1, feature_size)
        x = self.vlad_layer(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def initialize(self, model_path='Model/archi-videoResNET_PCA__VGGish_RVLAD64_2020-01-16_17-08-09_model.ckpt'):
        print("Pre-trained model loading is not implemented as PyTorch models are not saved in .ckpt format.")

# ---

## AudioNetwork (PyTorch)

class AudioNetwork(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioNetwork, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        if "RVLAD" in network_type.upper():
            self.vlad_layer = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        elif "VLAD" == network_type.upper():
            self.vlad_layer = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, audio_input):
        x = audio_input
        batch_size, num_frames, feature_size = x.shape
        x = x.view(-1, feature_size)
        x = self.vlad_layer(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def initialize(self, model_path='Model/archi2ResNET_PCA__VGGish_RVLAD64_2020-01-16_17-08-21_model.ckpt'):
        print("Pre-trained model loading is not implemented as PyTorch models are not saved in .ckpt format.")

# ---

## Archi3Prediction (PyTorch)

class Archi3Prediction(nn.Module):
    def __init__(self, dataset):
        super(Archi3Prediction, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.num_classes = dataset.num_classes

    def forward(self, video_logits, audio_logits):
        logits = 0.5 * video_logits + 0.5 * audio_logits
        return logits

# ---

## AudioVideoArchi4 (PyTorch)

class AudioVideoArchi4(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi4, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        # NetRVLAD의 output_dim이 512이므로, linear layer의 input_features도 512로 변경
        self.video_vlad = NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )
        self.audio_vlad = NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_audio"
        )

        self.video_dropout = nn.Dropout(p=0.5)
        self.video_fc = nn.Linear(512, self.num_classes)

        self.audio_dropout = nn.Dropout(p=0.5)
        self.audio_fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input, audio_input):
        x_video = video_input
        x_audio = audio_input
        batch_size, num_frames, feature_size = x_video.shape

        x_video = x_video.view(-1, feature_size)
        x_video = self.video_vlad(x_video)
        x_video = self.video_dropout(x_video)
        logits_video = self.video_fc(x_video)

        x_audio = x_audio.view(-1, feature_size)
        x_audio = self.audio_vlad(x_audio)
        x_audio = self.audio_dropout(x_audio)
        logits_audio = self.audio_fc(x_audio)

        logits = 0.5 * logits_video + 0.5 * logits_audio
        return logits, logits_video, logits_audio

# ---

## AudioVideoArchi5 (PyTorch)

class AudioVideoArchi5(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi5, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        if "RVLAD" in network_type.upper():
            self.video_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )

            self.audio_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        elif "VLAD" == network_type.upper():
            self.video_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.dropout = nn.Dropout(p=0.5)
        # NetRVLAD/NetVLAD의 output_dim은 512이므로, concatenate 후 1024가 됨
        self.fc = nn.Linear(512 + 512, self.num_classes)

    def forward(self, video_input, audio_input):
        x_video = video_input
        x_audio = audio_input
        batch_size, num_frames, feature_size = x_video.shape

        x_video = x_video.view(-1, feature_size)
        x_video = self.video_vlad(x_video)

        x_audio = x_audio.view(-1, feature_size)
        x_audio = self.audio_vlad(x_audio)

        x_video = self.dropout(x_video)
        x_audio = self.dropout(x_audio)

        x = torch.cat([x_video, x_audio], dim=1)
        logits = self.fc(x)
        return logits

    def initialize(self, model_path='TrainVlad/vlad-archi5-20secResNET_PCA__VGGish_VLAD512_2020-03-03_13-30-06_model.ckpt'):
        print("Pre-trained model loading is not implemented as PyTorch models are not saved in .ckpt format.")

# ---

## AudioVideoArchi6 (PyTorch)

class AudioVideoArchi6(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi6, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        self.video_vlad = NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        self.audio_vlad = NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_audio"
        )

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 + 512, self.num_classes)

    def forward(self, video_input, audio_input):
        x_video = video_input
        x_audio = audio_input
        batch_size, num_frames, feature_size = x_video.shape

        x_video = x_video.view(-1, feature_size)
        x_video = self.video_vlad(x_video)

        x_audio = x_audio.view(-1, feature_size)
        x_audio = self.audio_vlad(x_audio)

        x = torch.cat([x_video, x_audio], dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# ---

## AudioVideoArchi7 (PyTorch)

class AudioVideoArchi7(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi7, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        # concat된 feature_size는 512 + 512 = 1024
        self.vlad_layer = NetRVLAD(
            feature_size=1024,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=1024,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, video_input, audio_input):
        x_video = video_input
        x_audio = audio_input

        x = torch.cat([x_video, x_audio], dim=2)
        batch_size, num_frames, feature_size = x.shape

        x = x.view(-1, feature_size)
        x = self.vlad_layer(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# ---

## AudioVideoArchi8 (PyTorch)

class AudioVideoArchi8(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi8, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        # concat된 feature_size는 512 + 512 = 1024
        self.vlad_layer = NetRVLAD(
            feature_size=1024,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input, audio_input):
        x_video = video_input
        x_audio = audio_input

        x = torch.cat([x_video, x_audio], dim=2)
        batch_size, num_frames, feature_size = x.shape

        x = x.view(-1, feature_size)
        x = self.vlad_layer(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# ---

## Archi9Prediction (PyTorch)

class Archi9Prediction(nn.Module):
    def __init__(self, dataset):
        super(Archi9Prediction, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.num_classes = dataset.num_classes

    def forward(self, video_logits, audio_logits):
        video_predictions = torch.sigmoid(video_logits)
        audio_predictions = torch.sigmoid(audio_logits)
        predictions = video_predictions * audio_predictions

        logits = 0.5 * video_logits + 0.5 * audio_logits
        return logits, predictions

# ---

## AudioVideoArchi10 (PyTorch)

class AudioVideoArchi10(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi10, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        if "RVLAD" in network_type.upper():
            self.vlad_layer = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input):
        x = video_input
        batch_size, num_frames, feature_size = x.shape
        x = x.view(-1, feature_size)
        x = self.vlad_layer(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def initialize(self, model_path='Model/archi10ResNET_PCA__VGGish_RVLAD64_2020-01-27_13-49-52_model.ckpt'):
        print("Pre-trained model loading is not implemented as PyTorch models are not saved in .ckpt format.")

# ---

## AudioVideoArchi11 (PyTorch)

class AudioVideoArchi11(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        super(AudioVideoArchi11, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        if "RVLAD" in network_type.upper():
            self.vlad_layer = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, audio_input):
        x = audio_input
        batch_size, num_frames, feature_size = x.shape
        x = x.view(-1, feature_size)
        x = self.vlad_layer(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def initialize(self, model_path='Model/archi11ResNET_PCA__VGGish_RVLAD64_2020-01-27_17-04-57_model.ckpt'):
        print("Pre-trained model loading is not implemented as PyTorch models are not saved in .ckpt format.")

# ---

## Archi12Prediction (PyTorch)

class Archi12Prediction(nn.Module):
    def __init__(self, dataset):
        super(Archi12Prediction, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.num_classes = dataset.num_classes

    def forward(self, video_logits, audio_logits):
        logits = 0.5 * video_logits + 0.5 * audio_logits
        return logits

# ---

## Archi18Prediction (PyTorch)

class Archi18Prediction(nn.Module):
    def __init__(self, dataset):
        super(Archi18Prediction, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.num_classes = dataset.num_classes

    def forward(self, video_logits, audio_logits):
        video_predictions = torch.sigmoid(video_logits)
        audio_predictions = torch.sigmoid(audio_logits)
        predictions = video_predictions * audio_predictions

        logits = 0.5 * video_logits + 0.5 * audio_logits
        return logits, predictions


# ---
# CMGAArchi1
# early fusion # similar to AudioVideoArchi7 # more strictly following CMGA paper「Jiang, Ji(2022)」
class CMGAArchi1(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=False, VLAD_batch_norm=True):
        super(CMGAArchi1, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes
        self.cmga = CrossModalGatedAttention(feature_dim=512, num_heads=8, dropout=0.2)

        # VLAD layer for fused features
        if "RVLAD" in network_type.upper():
            self.vlad_layer = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_cmga"
            )
        elif "VLAD" == network_type.upper():
            self.vlad_layer = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_cmga"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input, audio_input):
      # each input: (B x N x D)
      # output: (B x N x D)

      # Early Fusion | Cross-modality Gated Attention
      fused_features = self.cmga(video_input, audio_input) # each input: (B x N x D) # ouptut: (B x N x D)
      _, _, feature_size = fused_features.shape
      x = fused_features.view(-1, feature_size)

      # VLAD Pooling (integrated)
      x = self.vlad_layer(x) # (B x output_dim=512)
      x = self.dropout(x)
      logits = self.fc(x) # (B x num_classes=5)

      return logits

# ---
# CMGAArchi2
# intermediate fusion # based on AudioVideoArchi5 # following action-spotting paper「Vanderplaetse, Dupont(2020)」

class CMGAArchi2(nn.Module):

    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=False, VLAD_batch_norm=True): # set context_gating as False
        super(CMGAArchi2, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        # Separate VLAD layers for video and audio
        if "RVLAD" in network_type.upper():
            self.video_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        elif "VLAD" == network_type.upper():
            self.video_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.cmga = CrossModalGatedAttention(feature_dim=512, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input, audio_input):
      # each input: (B x N x D)
      # output: (B x N x D)

      batch_size, num_frames, feature_size = video_input.shape

      # VLAD pooling (separate)
      x_video = video_input.view(-1, feature_size)
      video_vlad_output = self.video_vlad(x_video) # (B x output_dim=512)

      x_audio = audio_input.view(-1, feature_size)
      audio_vlad_output = self.audio_vlad(x_audio)  # (B x output_dim=512)

      # Intermediate Fusion | Cross-modality Gated Attention
      fused_features = self.cmga(video_vlad_output, audio_vlad_output) # each input: (B x output_dim=512) | output: (B x 1 x D=512)
      fused_features = fused_features.squeeze(1) # (B x D=512)
      x = self.dropout(fused_features)
      logits = self.fc(x) # (B x self.num_classes=5)

      return logits


# ---
# CMGAArchi3
# Early fusion # based on AudioVideoArchi7 # more strictly following CMGA paper「Jiang, Ji(2022)」while employing bottleneck unit from「Nagrani(2021)」

class CMGAArchi3(nn.Module):

    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=False, VLAD_batch_norm=True): # set context_gating as False
        super(CMGAArchi3, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes

        # Separate VLAD layers for video and audio
        if "RVLAD" in network_type.upper():
            self.video_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        elif "VLAD" == network_type.upper():
            self.video_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        else:
            raise ValueError("Unsupported network_type")

        self.cmga = CrossModalGatedBottleneckAttention(feature_dim=512, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, video_input, audio_input):
      # each input: (B x N x D)
      # output: (B x N x D)

      batch_size, num_frames, feature_size = video_input.shape

      # VLAD pooling (separate)
      x_video = video_input.view(-1, feature_size)
      video_vlad_output = self.video_vlad(x_video) # (B x output_dim=512)

      x_audio = audio_input.view(-1, feature_size)
      audio_vlad_output = self.audio_vlad(x_audio)  # (B x output_dim=512)

      # Intermediate Fusion | Cross-modality Gated Attention
      fused_features = self.cmga(video_vlad_output, audio_vlad_output) # each input: (B x output_dim=512) | output: (B x 1 x D=512)
      fused_features = fused_features.squeeze(1) # (B x D=512)
      x = self.dropout(fused_features)
      logits = self.fc(x) # (B x self.num_classes=5)

      return logits

# ---
# CMGAArchi4
# early fusion # similar to AudioVideoArchi7 # more strictly following CMGA paper「Jiang, Ji(2022)」

class CMGAArchi4(nn.Module): # CMGAArchi1 without VLAD Pooling
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=False, VLAD_batch_norm=True): # delete unused arguments if not cause error when removed <- all except dataset
        super(CMGAArchi4, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        self.num_classes = dataset.num_classes
        self.cmga = CrossModalGatedAttention(feature_dim=512, num_heads=8, dropout=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, self.num_classes) # 임시로 hard-coding

    def forward(self, video_input, audio_input):
      # each input: (B x N x D)
      # output: (B x num_classes)

      # Early Fusion | Cross-modality Gated Attention
      fused_features = self.cmga(video_input, audio_input) # each input: (B x N x D) # ouptut: (B x N x D)
      
      # Simple mean pooling
      x = fused_features.mean(dim=1) # (B x D)
      x = self.dropout(x)
      
      logits = self.fc(x) # (B x num_classes)

      return logits

# ---
# GMU
# intermediate fusion # following GMU CMGA John (2017)」  
  
class GMU(nn.Module):
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=False, VLAD_batch_norm=True): # set context_gating as False
        super(GMU, self).__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes
        # Separate VLAD layers for video and audio
        if "RVLAD" in network_type.upper():
            self.video_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        elif "VLAD" == network_type.upper():
            self.video_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            self.audio_vlad = NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
        else:
            raise ValueError("Unsupported network_type")
        self.fusion = GatedBimodalTorch(in_dim_v=512, in_dim_t=512, hid_dim=512)
        self.dropout = nn.Dropout(p=0.5)  # 0.6으로도 실험해보기
        self.fc = nn.Linear(512, self.num_classes)
    def forward(self, video_input, audio_input):
        B, N, D = video_input.shape
        x_video = video_input.reshape(-1, D)   # [B*N, D]
        v = self.video_vlad(x_video)           # [B, 512]
        x_audio = audio_input.reshape(-1, D)   # [B*N, D]
        a = self.audio_vlad(x_audio)           # [B, 512]
        fused, z = self.fusion(v, a)           # [B, 512]
        x = self.dropout(fused)                # [B, 512]
        logits = self.fc(x)                    # [B, num_classes]
        return logits
    
# ---
# LateFusionArchi with Transformer
# late fusion 


class LateFusionArchi(nn.Module):
    """
    RVLAD/VLAD로 비디오/오디오를 각각 [B, D]로 풀링한 뒤,
    [CLS, V, A] 3개 토큰에 대해 TransformerEncoder(1~2층)로 late fusion을 수행하는 모델.
    """
    def __init__(self, dataset, network_type="VLAD", VLAD_K=512, VLAD_gating=False, VLAD_batch_norm=True,
                 d_model=512, nhead=8, num_layers=2, dim_ff=1024, trans_dropout=0.1, cls_dropout=0.6, use_cls=True):
        super().__init__()
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.network_type = network_type
        self.VLAD_k = VLAD_K
        self.num_classes = dataset.num_classes
        self.d_model = d_model
        if "RVLAD" in network_type.upper():
            self.video_vlad = NetRVLAD(
                feature_size=d_model,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=d_model,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video",
            )
            self.audio_vlad = NetRVLAD(
                feature_size=d_model,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=d_model,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio",
            )
        elif "VLAD" == network_type.upper():
            self.video_vlad = NetVLAD(
                feature_size=d_model,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=d_model,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video",
            )
            self.audio_vlad = NetVLAD(
                feature_size=d_model,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=d_model,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio",
            )
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
        self.fusion = LateFusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=trans_dropout,
            use_cls=use_cls,
        )
        self.dropout = nn.Dropout(p=cls_dropout)
        self.fc = nn.Linear(d_model, self.num_classes)
    def forward(self, video_input, audio_input):
        """
        video_input, audio_input: [B, N, D] (예: D=512)
        반환 logits: [B, num_classes]
        """
        # VLAD pooling (각각 [B, D])
        video_vec = self.video_vlad(video_input)
        audio_vec = self.audio_vlad(audio_input)
        # Late fusion via Transformer over [CLS, V, A]
        fused = self.fusion(video_vec, audio_vec)  # [B, D]
        x = self.dropout(fused)
        logits = self.fc(x)  # [B, num_classes]
        return logits