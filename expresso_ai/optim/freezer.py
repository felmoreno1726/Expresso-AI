import math

from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torch.nn.parallel.data_parallel import DataParallel

class Freezer():


    def __init__(self, model, optimizer = None, defreeze_last_n = 1, skip_empty_layers=True):
        """
        @param model (torch.nn.model): The training model to be frozen and defreezed progressively.
        @param optimizer (torch.nn.optimizer): Object that optimizes the training objective by modifying model weights.
        @param defreeze_last_n (int): Number of layers to be defreezed from the start.
        @param skip_empty_layers (Bool): If true, layers that contain no paramters are skipped.
        """
        #check that inputs have valid types
        if optimizer is not None:
            if not isinstance(optimizer, Optimizer):
                raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        if not isinstance(model, Module):
            raise TypeError(f"{type(model).__name__} is not a Torch Module")
        #initialize freezer
        if isinstance(model, DataParallel):
            print("model is DataParallel instance")
            self.model = model.module
        else:
            self.model = model
        self.optimizer = optimizer
        self._step_count = 0
        self._layer_count = 0
        self.skip_empty_layers = skip_empty_layers
        #freeze model
        self.freeze_model(model)
        #defreeze last n layers immediately
        for i in range(defreeze_last_n):
            #self.step(skip_empty_layers=False)
            self.defreeze_next_layer(skip_empty_layers=False)


    def set_optimizer(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer


    def set_optimization_params(self, optimization_params):
        self.optimization_params = optimization_params


    def get_optimization_params(self):
        return {**self.optimization_params}


    def step(self, skip_empty_layers=None):
        self._step_count += 1
        self.defreeze_next_layer(skip_empty_layers=skip_empty_layers)


    def defreeze_next_layer(self, skip_empty_layers=None):
        self._layer_count += 1
        self.defreeze_last_n_layer(self._layer_count, skip_empty_layers if skip_empty_layers is not None else self.skip_empty_layers)


    def defreeze_last_n_layer(self, layer_position, skip_empty_layers):
        """
        This method takes the last n layer, sets the weights require_gradients to True, and adds the respective paramters to the optimizer if optimizer is defined
        @param layer_position (int): position of layer to be defrozen.
        """
        #import pdb; pdb.set_trace()
        #model_layers = list(self.model.children())
        model_layers = list(self.model.modules())
        num_layers = len(model_layers)
        layer_idx = num_layers - layer_position
        if layer_idx < 0:
            print("Model is thawed!")
            return
        else:
            layer = model_layers[layer_idx]
            print(f"Defreezing layer {layer_idx}: ", layer)
            #defreeze layer
            self.defreeze_weights(layer)
            #add layer to optimizer
            if self.optimizer is not None:
                #print("adding layer to optimizer parameters")
                #print("before: ", self.optimizer.param_groups)
                if len(list(layer.parameters())) == 0:
                    print("Layer has no parameters")
                    if skip_empty_layers:
                        print("Skipping current layer to defreeze next")
                        #self.step()
                        self.defreeze_next_layer()
                else:
                    #adds parameters of the next layers to the optimizer
                    try:
                        self.optimizer.add_param_group({"params": layer.parameters()})
                        #print("after: ", self.optimizer.param_groups)
                    except Exception as e:
                        print("Exception: ", e)
                        self.defreeze_next_layer()


    @staticmethod
    def freeze_model(model):
        for parameter in model.parameters():
            parameter.requires_grad = False


    @staticmethod
    def defreeze_weights(layer):
        for parameter in layer.parameters():
            parameter.requires_grad = True
   

    @staticmethod
    def freeze_weights(layer):
        for parameter in layer.parameters():
            parameter.requires_grad = False


class MultiplicativeFreezer(Freezer):
    """
    Freezer in initially freezes all model layers and progressively defreezes them
    """
    def __init__(self, model, optimizer, step_size = 1, gamma=1, last_epoch=-1):
        self.model = model
        self.optimizer = optimizer
        self.T_mult = T_mult
        assert last_epoch >= -1, f"Last_epoch parameter equals {last_epoch} but must be greater >= -1"
        self.last_epoch = last_epoch


    def step(self, epoch=None):
        if epoch is None and last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = last_epoch + 1
            self.T_cur = self.T_cur + 1
            #check if gone on to next cycle
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
            self.last_epoch = math.floor(epoch)
        raise NotImplementedError("method has not yet been fully implemented")
        return


class StepFreezer(Freezer):

    def __init__(self, model, optimizer = None, step_size=1, defreeze_last_n = 1, skip_empty_layers=True):
        """
        @param model (torch.nn.model): The training model to be frozen and defreezed progressively.
        @param optimizer (torch.nn.optimizer): Object that optimizes the training objective by modifying model weights.
        @param step_size (int): Sets the number of steps that need to be called for the following layer to be defreezed
        @param defreeze_last_n (int): Number of layers to be defreezed from the start.
        @param skip_empty_layers (Bool): If true, layers that contain no paramters are skipped.
        """
        self.step_size = step_size
        self.layer_count = 0
        super(StepFreezer, self).__init__(model, optimizer, defreeze_last_n, skip_empty_layers)
        
    def step(self, skip_empty_layers=None):
        self._step_count += 1
        if self._step_count % self.step_size == 0:
            self.defreeze_next_layer(skip_empty_layers=skip_empty_layers)

class TwoStepFreezer(Freezer):

    def __init__(self, model, optimizer = None, initial_step=1, step_size=1, defreeze_last_n = 1, skip_empty_layers=True):
        """
        @param model (torch.nn.model): The training model to be frozen and defreezed progressively.
        @param optimizer (torch.nn.optimizer): Object that optimizes the training objective by modifying model weights.
        @param step_size (int): Sets the number of steps that need to be called for the following layer to be defreezed
        @param defreeze_last_n (int): Number of layers to be defreezed from the start.
        @param skip_empty_layers (Bool): If true, layers that contain no paramters are skipped.
        """
        self.initial_step = initial_step
        print("initial step? ", self.initial_step)
        self.step_size = step_size
        super(TwoStepFreezer, self).__init__(model, optimizer, defreeze_last_n, skip_empty_layers)
        
    def step(self, skip_empty_layers=None):
        self._step_count += 1
        if (self._step_count == self.initial_step) or (self._step_count > self.initial_step and self._step_count % self.step_size == 0):
            self.defreeze_next_layer(skip_empty_layers=skip_empty_layers)


class MultiStepFreezer(Freezer):
    """
    @param model (torch.nn.model): The training model to be frozen and defreezed progressively.
    @param optimizer (torch.nn.optimizer): Object that optimizes the training objective by modifying model weights.
    @param milestones (int[]): Sets the epochs at which a layer must be "thawed"
    @param defreeze_last_n (int): Number of layers to be defreezed from the start.
    @param skip_empty_layers (Bool): If true, layers that contain no paramters are skipped.
    """

    def __init__(self, model, optimizer = None, milestones=[], defreeze_last_n = 1, skip_empty_layers=True):
        self.milestones = milestones
        super(StepFreezer, self).__init__(model, optimizer, defreeze_last_n, skip_empty_layers)
        
    def step(self, skip_empty_layers=None):
        self._step_count += 1
        if self._step_count in self.milestones:
            self.defreeze_next_layer(skip_empty_layers=skip_empty_layers)

