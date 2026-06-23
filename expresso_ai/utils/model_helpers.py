import os, copy
from collections import OrderedDict

import torch

def robust_load_weights(model, state_dict):
    """
    Function loads a state_dict to a model in a robust fail safe way. It asserts that the model is loaded such that it can be done for either a DataParallel model or a non-DataParallel model
    @param model (nn.Module): model to load state dictionary to.
    @param state_dict (dict): state dictionary to load into the model.
    """
    try:#model is DataParallel
        model.module.load_state_dict(state_dict)
    except:
        #model is not DataParallel
        try:
            model.load_state_dict(state_dict)
        except:
            #model weights were originaly saved with a DataParallel model
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] #remove word "module." from state key name
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)


def load_checkpoint(file_path, load_best=False):
    print('file_path: ', file_path)
    if not os.path.isfile(file_path):
        print('checkpoint path is not a file!')
        return None
    #state = torch.load(file_path, map_location=default_device)
    state = torch.load(file_path, map_location='cpu')
    if load_best:
        state = state["best"]
    print('loaded pytorch model')
    return state

def save_checkpoint(state, is_best, filename, directory_path='./weights'):
    """
    Saves the state of the model to a filestructure which keeps track of both the current state and the best state of the model.
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, 0o777)
    file_path = os.path.join(directory_path, filename)
    if is_best:
        print('saving best to checkpoint')
        #make a copy of state and save it as best
        best_state = copy.deepcopy(state)
        state['best'] = best_state
    else:
        #reload the previous best
        state['best'] = load_checkpoint(file_path, load_best=True)
    print('state modified with best: ', state.keys())
    torch.save(state, file_path)
