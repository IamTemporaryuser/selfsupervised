from .cls_head import LinearClsHead
from .pred_mlp import Pred_MLP
from .saliency_fc import Saliency_FC
from .saliencycls_head import SaliencyCLSHead
from .parametric_head import ParametricHead
from .simsiam_head import SimsiamHead
__all__ = ['LinearClsHead', 'Pred_MLP', 'Saliency_FC', 'SaliencyCLSHead', 'ParametricHead', "SimsiamHead"]