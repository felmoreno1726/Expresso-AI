
import torch

from .inception_resnet_v1 import InceptionResnetV1


class VideoFaceNet(InceptionResnetV1):

    """
    Class adds a wrapper around the InceptionResnetV1 implementation of FaceNet to allow for the model to be used with videos of shape (3 x T x H x W)
    """

    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super(VideoFaceNet, self).__init__(
                pretrained=pretrained,
                classify=classify,
                num_classes=num_classes,
                dropout_prob=dropout_prob,
                device=device)
        #define the output features of the model
        if classify and self.num_classes is not None:
            self.features_out = num_classes
        else:
            self.features_out = 512


    def forward(self, x):
        batch_size = x.shape[0]#Batch
        signal_length = x.shape[2]#T
        #from [Batch x C x T x W x H] -> [Batch x T x (C x W x H)]
        x = torch.transpose(x, 2, 1).contiguous()
        #cluster channels and temporal axis into 1
        x = x.view((-1,) + x.shape[2:])
        #import pdb; pdb.set_trace()
        x = super().forward(x)
        #Recover [Batch x C x T x W x H]
        #x = x.view((batch_size, self.features_out, signal_length, *x.shape[2:]))
        x = x.view((batch_size, self.features_out, signal_length, 1, 1))
        return x
