import math

from ignite.exceptions import NotComputableError

from .clustered_mse import ClusteredMSE

__all__ = ["ClusteredRMSE"]

class ClusteredRMSE(ClusteredMSE):


    def compute(self):
        clustered_mse = super(ClusteredRMSE, self).compute()
        if clustered_mse is not None:
            if clustered_mse >= 0:
                return math.sqrt(clustered_mse)
        raise NotComputableError
