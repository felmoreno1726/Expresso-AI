

import torch
import torch.nn as nn
from torchvision.models.video.resnet import BasicBlock, Conv3DSimple, Bottleneck
#from facenet_pytorch import InceptionResnetV1

from .transfer_video_resnet import TransferVideoResNet
from .facenet_video import VideoFaceNet


class FaceNetRMC_Simple_1(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Simple_1, self).__init__(
                block=BasicBlock,
                conv_makers=[Conv3DSimple],
                layers = [3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Simple_2(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Simple_2, self).__init__(
                block=BasicBlock,
                conv_makers=[Conv3DSimple] * 2,
                layers = [3, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Simple_3(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Simple_3, self).__init__(
                block=BasicBlock,
                conv_makers=[Conv3DSimple] * 3,
                layers = [3, 4, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Simple_4(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Simple_4, self).__init__(
                block=BasicBlock,
                conv_makers=[Conv3DSimple] * 4,
                layers = [3, 4, 6, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Simple_5(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Simple_5, self).__init__(
                block=BasicBlock,
                conv_makers=[Conv3DSimple] * 4,
                layers = [3, 4, 23, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Bottleneck_1(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Bottleneck_1, self).__init__(
                block=Bottleneck,
                conv_makers=[Conv3DSimple],
                layers = [3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Bottleneck_2(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Bottleneck_2, self).__init__(
                block=Bottleneck,
                conv_makers=[Conv3DSimple] * 2,
                layers = [3, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Bottleneck_3(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Bottleneck_3, self).__init__(
                block=Bottleneck,
                conv_makers=[Conv3DSimple] * 3,
                layers = [3, 4, 3],#[3,6,3]
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Bottleneck_4(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Bottleneck_4, self).__init__(
                block=Bottleneck,
                conv_makers=[Conv3DSimple] * 4,
                layers = [3, 4, 6, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )


class FaceNetRMC_Bottleneck_5(TransferVideoResNet):

    def __init__(self, num_classes=400, zero_init_residual=False, **kwargs):
        """
        """
        facenet_arg_names = {"pretrained", "classify", "num_classes", "dropout"}
        facenet_args = dict(filter(lambda x: x[0] in facenet_arg_names, kwargs.items()))
        facenet_video_model = VideoFaceNet(**facenet_args)
        inplanes = facenet_video_model.features_out
        def facenet_wrapper():
            return facenet_video_model
        super(FaceNetRMC_Bottleneck_5, self).__init__(
                block=Bottleneck,
                conv_makers=[Conv3DSimple] * 4,
                layers = [3, 4, 23, 3],
                stem = facenet_wrapper,
                #stem = lambda : None,#NOTE: stem is set later on. This passes a Null stem constructor
                num_classes = num_classes,
                inplanes=inplanes,
                zero_init_residual=zero_init_residual
        )
