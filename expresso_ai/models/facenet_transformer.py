

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNetTransformer(nn.Module):

    def __init__(self, **kwargs):
        """
        """
        super(DeepFaceTransformer, self).__init__()
        facenet_arg_names = {"pretrained", "classify", "num_classes"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        self.facenet_model = InceptionResnetV1(**facenet_args)
        #initialize subsequent 3D CNN blocks
        ###TODO:

    def forward(self, x):
        raise NotImplementedError("Forward not yet implemented")


