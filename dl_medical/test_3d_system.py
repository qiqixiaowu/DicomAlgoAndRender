"""快速验证 3D 配准系统所有模块"""
import sys
sys.path.insert(0, '.')

# 测试模型导入
from models import (
    VoxelMorph3D, VoxelMorph3DDiff, SpatialTransformer3D, VecInt,
    NCCLoss3D, MSELoss3D, LNCCLoss3D,
    GradientLoss3D, BendingEnergyLoss3D, JacobianLoss, RegistrationLoss3D,
)
print('[OK] 模型导入成功')

# 测试数据导入
from data import (
    RegistrationDataset3D, NIfTIDataset3D, PairedNIfTIDataset3D,
    SyntheticLungData3D, SyntheticLiverData3D, SyntheticRegData3D,
    Compose3D, get_default_transforms_3d, get_val_transforms_3d,
)
print('[OK] 数据导入成功')

# 测试指标导入
from utils.metrics3d import (
    jacobian_determinant_3d, folding_ratio_3d, jacobian_stats_3d,
    dice_3d, ssim_3d, psnr_3d, evaluate_registration_3d,
    target_registration_error,
)
print('[OK] 指标导入成功')

# 测试模型前向传播
import torch
print('\n--- 测试模型前向传播 ---')
device = torch.device('cpu')
fixed = torch.randn(1, 1, 32, 32, 32)
moving = torch.randn(1, 1, 32, 32, 32)

# VoxelMorph3D
model1 = VoxelMorph3D(image_size=(32, 32, 32), enc_features=[8, 16, 32], dec_features=[32, 16, 8])
out1 = model1(fixed, moving)
print(f'VoxelMorph3D:      warped={out1[0].shape}, flow={out1[1].shape}')

# VoxelMorph3DDiff
model2 = VoxelMorph3DDiff(image_size=(32, 32, 32), enc_features=[8, 16, 32], dec_features=[32, 16, 8], vecint_steps=7)
out2 = model2(fixed, moving)
print(f'VoxelMorph3DDiff:  warped={out2[0].shape}, flow={out2[1].shape}, velocity={out2[2].shape}')

# 测试损失
print('\n--- 测试损失 ---')
reg_loss = RegistrationLoss3D(
    sim_loss=LNCCLoss3D((5, 9)),
    smooth_loss=BendingEnergyLoss3D(),
    sim_weight=1.0, smooth_weight=0.5, jac_weight=0.1,
)
total, ld = reg_loss(fixed, out2[0], out2[1])
print(f'组合损失: {total.item():.4f}  {ld}')

# 测试数据生成
print('\n--- 测试数据生成 ---')
lung_gen = SyntheticLungData3D((64, 64, 64))
f, m = lung_gen.generate_one()
print(f'肺部数据: fixed={f.shape}, moving={m.shape}')

liver_gen = SyntheticLiverData3D((64, 64, 64))
f, m = liver_gen.generate_one()
print(f'肝部数据: fixed={f.shape}, moving={m.shape}')

# 测试数据集
print('\n--- 测试数据集 ---')
ds = RegistrationDataset3D(n_samples=3, volume_size=(32, 32, 32), data_type='lung',
                           transform=get_default_transforms_3d())
sample = ds[0]
print(f'数据集样本: fixed={sample["fixed"].shape}, moving={sample["moving"].shape}')

# 测试指标
print('\n--- 测试指标 ---')
flow = torch.randn(1, 3, 32, 32, 32) * 0.05
print(f'Jacobian stats: {jacobian_stats_3d(flow)}')
print(f'Folding ratio: {folding_ratio_3d(flow):.4f}')
print(f'SSIM: {ssim_3d(fixed, fixed + torch.randn_like(fixed)*0.1):.4f}')
print(f'PSNR: {psnr_3d(fixed, fixed + torch.randn_like(fixed)*0.1):.4f}')

print('\n' + '='*50)
print('所有测试通过! 3D 配准系统就绪!')
print('='*50)
