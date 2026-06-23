import os

import torch
from torch import nn

from ..utils.model_helpers import robust_load_weights

def initialize_model(model_name, *args, **kwargs):
    """
    Function intializes a model
    @param model_name (str) the name of the model to be initialized. Choose from facenet, facenet_rmc, facenet_2plus1d, r3d_18, mc3_18, r2plus1d_18, r3d, r2p1d, resnet, densenet, preresnet, resnext, wideresnet
    @param args additional positional arguments to pass to model constructor 
    @kwargs additional keyword arguments to pass to model constructor
    @returns model_instance, dimensions: The pytorch model and the (width, height) expected for the model's input
    """
    if model_name == "facenet":
        from facenet_pytorch import InceptionResnetV1
        #from .inception_resnet_v1 import InceptionResnetV1
        dimensions = (150, 150)
        channels = 3
        #NOTE: Using pretrained_dataset instead of pretrained argument passed to r3d_18
        facenet_args = {"dropout_prob", "pretrained"}
        default_arguments = {
                "num_classes": 1, 
                "classify": True,
                #"pretrained": "casia-webface"###TODO: try vggface2
        }
        arguments = dict(filter(lambda x : x[0] in facenet_args, kwargs.items()))
        arguments.update(default_arguments)
        model = InceptionResnetV1(**arguments)
    elif model_name == "facenet_rmc":
        #from .facenet_rmc import FaceNetRMC
        from .facenet_rmc import FaceNetRMC_Simple_1 as facenet_rmc_model
        dimensions = (150, 150)
        channels = 3
        #NOTE: Using pretrained_dataset instead of pretrained argument passed to r3d_18
        facenet_args = {"dropout_prob", "pretrained"}
        default_arguments = {
                "classify": False,###NOTE: setting to False avoids using the final layer
                "zero_init_residual": False,
                "num_classes": 1,
                "pretrained": "casia-webface",###TODO: try vggface2
        }
        arguments = dict(filter(lambda x : x[0] in facenet_args, kwargs.items()))
        arguments.update(default_arguments)
        model = facenet_rmc_model(**arguments)
    elif model_name == "facenet_2plus1d":
        from .facenet_2plus1d import FaceNet2Plus1D
        dimensions = (150, 150)
        channels = 3
        #facenet_args = {"pretrained"}
        default_arguments = {
                "classify": False,
                "pretrained": "casia-webface",###TODO: try vggface2

        }
        arguments = dict(filter(lambda x : x[0] in facenet_args, kwargs.items()))
        arguments.update(default_arguments)
        model = FaceNet2Plus1D(**arguments)
    else:
        #model is a torchvision model
        #only import these if neccessary as they require specific torchvision versions
        if model_name == "r3d_18":
            from torchvision.models.video import r3d_18
            torchvision_args = {"pretrained", "progress"}
            dimensions = (112, 112)
            channels = 3
            arguments = dict(filter(lambda x : x[0] in torchvision_args, kwargs.items()))
            model = r3d_18(**arguments)
        elif model_name == "mc3_18":
            from torchvision.models.video import mc3_18
            torchvision_args = {"pretrained", "progress"}
            dimensions = (112, 112)
            channels = 3
            arguments = dict(filter(lambda x : x[0] in torchvision_args, kwargs.items()))
            model = mc3_18(**arguments)
        elif model_name == "r2plus1d_18":
            from torchvision.models.video import r2plus1d_18
            torchvision_args = {"pretrained", "progress"}
            dimensions = (112, 112)
            channels = 3
            arguments = dict(filter(lambda x : x[0] in torchvision_args, kwargs.items()))
            model = r2plus1d_18(**arguments)

        # 3D ResNets paper. 2018
        elif model_name == "resnet":
            from . import resnet
            argument_names = {"model_depth"}
            default_arguments = {
                    "n_classes": 400,
                    "n_input_channels": 3,
                    "shortcut_type": 'A',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "widen_factor": 1.0
            }
            dimensions = (112, 112)
            channels = 3
            model_depth = kwargs['model_depth']
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            arguments["shortcut_type"] = 'A' if model_depth <= 34 else 'B'
            default_arguments.update(arguments)
            model = resnet.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}-{model_depth}-kinetics.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        elif model_name == "densenet":
            from . import densenet
            argument_names = {"model_depth"}
            default_arguments = {
                    "n_classes": 400,
                    "n_input_channels": 3,
                    "shortcut_type": 'B',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "widen_factor": 1.0
            }
            dimensions = (112, 112)
            channels = 3
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            default_arguments.update(arguments)
            model = densenet.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}-{model_depth}-kinetics.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        elif model_name == "preresnet":
            from . import pre_act_resnet
            argument_names = {"model_depth"}
            default_arguments = {
                    "n_classes": 400,
                    "n_input_channels": 3,
                    "shortcut_type": 'B',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "widen_factor": 1.0
            }
            dimensions = (112, 112)
            channels = 3
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            default_arguments.update(arguments)
            model = pre_act_resnet.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}-{model_depth}-kinetics.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        elif model_name == "resnet2p1d":
            from . import resnet2p1d
            default_arguments = {
                    "n_classes": 400,
                    "n_input_channels": 3,
                    "shortcut_type": 'B',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "widen_factor": 1.0
            }
            dimensions = (112, 112)
            channels = 3
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            default_arguments.update(arguments)
            model = resnet2p1d.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}-{model_depth}-kinetics.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        elif model_name == "resnext":
            from . import resnext
            argument_names = {"model_depth"}
            default_arguments = {
                    "cardinality": 32,
                    "n_classes": 400,
                    "n_input_channels": 3,
                    "shortcut_type": 'B',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
            }
            dimensions = (112, 112)
            channels = 3
            model_depth = kwargs['model_depth']
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            arguments["shortcut_type"] = 'A' if model_depth <= 34 else 'B'
            default_arguments.update(arguments)
            model = resnext.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}-{model_depth}-kinetics.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        elif model_name == "wideresnet":
            from . import wide_resnet
            argument_names = {"model_depth"}
            default_arguments = {
                    "n_classes": 400,
                    "n_input_channels": 3,
                    "shortcut_type": 'A',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "k": 2
            }
            dimensions = (112, 112)
            channels = 3
            model_depth = kwargs['model_depth']
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            arguments["shortcut_type"] = 'A' if model_depth <= 34 else 'B'
            default_arguments.update(arguments)
            model = wide_resnet.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}-{model_depth}-kinetics.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        # 2020 3D ResNet
        elif model_name == "r3d":
            from . import resnet
            argument_names = {"model_depth"}
            default_arguments = {
                    "n_classes": 700,
                    "n_input_channels": 3,
                    "shortcut_type": 'B',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "widen_factor": 1.0,
            }
            dimensions = (112, 112)
            channels = 3
            model_depth = kwargs['model_depth']
            pretrained_dataset = kwargs['pretrained_dataset']
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            #arguments["shortcut_type"] = 'A' if model_depth <= 34 else 'B'
            #define number of classes depending on the pretrained datasest(s)
            n_classes = (700) * ('K' in pretrained_dataset) + (339) * ('M' in pretrained_dataset) + (100) * ('S' in pretrained_dataset)
            arguments["n_classes"] = n_classes
            default_arguments.update(arguments)
            model = resnet.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}{model_depth}_{pretrained_dataset}_200ep.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        elif model_name == "r2p1d":
            from . import resnet2p1d
            argument_names = {"model_depth"}
            default_arguments = {
                    "n_classes": 700,
                    "n_input_channels": 3,
                    "shortcut_type": 'B',
                    "conv1_t_size": 7,
                    "conv1_t_stride":1,
                    "no_max_pool": False,
                    "widen_factor": 1.0,
            }
            dimensions = (112, 112)
            channels = 3
            model_depth = kwargs['model_depth']
            pretrained_dataset = kwargs['pretrained_dataset']
            arguments = dict(filter(lambda x : x[0] in argument_names, kwargs.items()))
            #arguments["shortcut_type"] = 'A' if model_depth <= 34 else 'B'
            #define number of classes depending on the pretrained datasest(s)
            n_classes = (700) * ('K' in pretrained_dataset) + (339) * ('M' in pretrained_dataset) + (100) * ('S' in pretrained_dataset)
            arguments["n_classes"] = n_classes
            default_arguments.update(arguments)
            model = resnet2p1d.generate_model(**default_arguments)
            if kwargs['pretrained']:
                pretrained_weights_path = os.path.join(kwargs['pretrained_weights_path'], f'{model_name}{model_depth}_{pretrained_dataset}_200ep.pth')
                pretrained_weights = torch.load(pretrained_weights_path, map_location='cpu')
                robust_load_weights(model, pretrained_weights['state_dict'])

        else:
            raise Exception("Model name {} is not known".format(model_name))
        #then replace the final layer of the model to match num_classes in our problem
        adapt_final_layer(model, kwargs["num_classes"])
    return model, dimensions, channels


def adapt_final_layer(model, num_classes):
    """
    Function replaces (mutates) the model by replacing the last layer 
    with a Linear layer with out_features = num_classes
    return (void)
    """
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return
