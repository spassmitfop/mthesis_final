from .masked_dqn import *
from .frostbite import *
from .masked_dqn_variants import *
from .masked_dqn_mutiple import *
from .obj_plus_masks import *
from .obj_extensions import *
# aliales to match the names in the paper
from .masked_dqn import (
    BinaryMaskWrapper as BinaryMasksWrapper,
    ObjectTypeMaskWrapper as ClassMasksWrapper,
    PixelMaskWrapper as ObjectMasksWrapper,
    ObjectTypeMaskPlanesWrapper as PlanesWrapper
)