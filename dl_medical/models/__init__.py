from .unet2d import UNet2D
from .unet3d import UNet3D
from .voxelmorph import VoxelMorph2D, SpatialTransformer
from .losses import (
    DiceLoss, DiceCELoss,
    NCCLoss, GradientLoss,
    BendingEnergyLoss,
)

# 3D 配准模型与损失
from .voxelmorph3d import (
    SpatialTransformer3D,
    VecInt,
    VoxelMorph3D,
    VoxelMorph3DDiff,
)
from .losses3d import (
    NCCLoss3D,
    MSELoss3D,
    LNCCLoss3D,
    GradientLoss3D,
    BendingEnergyLoss3D,
    JacobianLoss,
    RegistrationLoss3D,
)

__all__ = [
    # 2D
    "UNet2D", "UNet3D",
    "VoxelMorph2D", "SpatialTransformer",
    "DiceLoss", "DiceCELoss",
    "NCCLoss", "GradientLoss", "BendingEnergyLoss",
    # 3D
    "SpatialTransformer3D", "VecInt",
    "VoxelMorph3D", "VoxelMorph3DDiff",
    "NCCLoss3D", "MSELoss3D", "LNCCLoss3D",
    "GradientLoss3D", "BendingEnergyLoss3D",
    "JacobianLoss", "RegistrationLoss3D",
]
