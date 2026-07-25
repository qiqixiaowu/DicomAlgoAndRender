"""
3D 配准损失函数集合

损失组成
========
  Total Loss = w_sim · L_sim + w_smooth · L_smooth + w_jac · L_jac

1. L_sim    : 相似性损失 (NCC3D / MSE / LNCC3D)
2. L_smooth : 形变场正则化 (梯度 / 弯曲能量)
3. L_jac    : Jacobian 负值惩罚 (防止折叠)

工程要点
========
- LNCC3D 使用局部窗口 NCC, 对亮度变化更鲁棒
- Jacobian 惩罚直接约束 det(J) > 0, 保证拓扑
- 弯曲能量比一阶梯度更强, 适合大形变场景
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  相似性损失
# ============================================================

class NCCLoss3D(nn.Module):
    """
    3D 归一化互相关损失

    NCC(I, J) = Σ[(I - μ_I)(J - μ_J)] / √[Σ(I-μ_I)² · Σ(J-μ_J)²]

    使用局部窗口计算 (通过 3D 卷积实现), 捕捉局部结构相似性。
    """

    def __init__(self, window_size: int = 9):
        super().__init__()
        self.window_size = window_size

    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        fixed, warped : (B, 1, D, H, W)
        返回: -NCC (最小化 → 最大化相似度)
        """
        w = self.window_size
        pad = w // 2

        # 均匀卷积核
        kernel = torch.ones(
            1, 1, w, w, w, device=fixed.device, dtype=fixed.dtype
        ) / (w * w * w)

        # 局部均值
        I_mean = F.conv3d(fixed, kernel, padding=pad)
        J_mean = F.conv3d(warped, kernel, padding=pad)

        # 去均值
        I_c = fixed - I_mean
        J_c = warped - J_mean

        # 局部互相关
        cross = F.conv3d(I_c * J_c, kernel, padding=pad)
        I_var = F.conv3d(I_c * I_c, kernel, padding=pad)
        J_var = F.conv3d(J_c * J_c, kernel, padding=pad)

        ncc = cross / (torch.sqrt(I_var * J_var) + 1e-5)
        return -ncc.mean()


class MSELoss3D(nn.Module):
    """3D 均方误差 (同模态配准)"""

    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fixed, warped)


class LNCCLoss3D(nn.Module):
    """
    局部归一化互相关 (Local NCC) — 多窗口版本

    在多个尺度上计算 NCC, 对不同大小的结构都有响应。
    适合肺部/肝部等包含多尺度结构的场景。
    """

    def __init__(self, window_sizes: tuple[int, ...] = (5, 9)):
        super().__init__()
        self.ncc_losses = nn.ModuleList([NCCLoss3D(w) for w in window_sizes])

    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return sum(ncc(fixed, warped) for ncc in self.ncc_losses) / len(self.ncc_losses)


# ============================================================
#  形变场正则化
# ============================================================

class GradientLoss3D(nn.Module):
    """
    3D 形变场一阶梯度平滑正则化

    L = Σ ‖∇φ‖² = Σ (‖∂φ/∂z‖² + ‖∂φ/∂y‖² + ‖∂φ/∂x‖²)
    """

    def __init__(self, penalty: str = "l2"):
        super().__init__()
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        flow : (B, 3, D, H, W)
        """
        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty == "l2":
            return (dz ** 2).mean() + (dy ** 2).mean() + (dx ** 2).mean()
        else:
            return dz.abs().mean() + dy.abs().mean() + dx.abs().mean()


class BendingEnergyLoss3D(nn.Module):
    """
    3D 弯曲能量正则化 (二阶导数)

    L_be = Σ (∂²φ/∂xᵢ²)² + 2·Σ(∂²φ/∂xᵢ∂xⱼ)²

    比一阶梯度更强: 允许线性变化, 惩罚非线性弯曲。
    适合肝部等需要大形变但需保持平滑的场景。
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        flow : (B, 3, D, H, W)
        """
        # 二阶差分
        d2z = flow[:, :, 2:, :, :] + flow[:, :, :-2, :, :] - 2 * flow[:, :, 1:-1, :, :]
        d2y = flow[:, :, :, 2:, :] + flow[:, :, :, :-2, :] - 2 * flow[:, :, :, 1:-1, :]
        d2x = flow[:, :, :, :, 2:] + flow[:, :, :, :, :-2] - 2 * flow[:, :, :, :, 1:-1]

        # 交叉项
        dzy = (flow[:, :, 1:, 1:, :] + flow[:, :, :-1, :-1, :]
               - flow[:, :, 1:, :-1, :] - flow[:, :, :-1, 1:, :])
        dzx = (flow[:, :, 1:, :, 1:] + flow[:, :, :-1, :, :-1]
               - flow[:, :, 1:, :, :-1] - flow[:, :, :-1, :, 1:])
        dyx = (flow[:, :, :, 1:, 1:] + flow[:, :, :, :-1, :-1]
               - flow[:, :, :, 1:, :-1] - flow[:, :, :, :-1, 1:])

        return (
            (d2z ** 2).mean() + (d2y ** 2).mean() + (d2x ** 2).mean()
            + 2.0 * ((dzy ** 2).mean() + (dzx ** 2).mean() + (dyx ** 2).mean())
        )


# ============================================================
#  Jacobian 折叠惩罚
# ============================================================

class JacobianLoss(nn.Module):
    """
    Jacobian 行列式负值惩罚

    原理
    ====
    形变场 φ 的 Jacobian 行列式 det(J) 反映局部体积变化:
    - det(J) > 0 : 正常 (无折叠)
    - det(J) ≤ 0 : 折叠! (拓扑被破坏)

    惩罚: L_jac = Σ max(0, -det(J))²

    只惩罚负值, 不惩罚大的正值 (允许膨胀)。
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        flow : (B, 3, D, H, W) — 位移场 (dz, dy, dx)
        """
        # 计算 Jacobian 行列式
        jac_det = self._jacobian_determinant(flow)
        # 惩罚负值
        return (torch.clamp(-jac_det, min=0.0) ** 2).mean()

    @staticmethod
    def _jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
        """
        计算 3D Jacobian 行列式

        flow: (B, 3, D, H, W), 通道顺序 (dz, dy, dx)

        J = | ∂φz/∂z  ∂φz/∂y  ∂φz/∂x |
            | ∂φy/∂z  ∂φy/∂y  ∂φy/∂x |
            | ∂φx/∂z  ∂φx/∂y  ∂φx/∂x |

        det(J) = (Dz·Dy·Dx + Dy_x·Dx_z·Dz_y + Dx_y·Dz_x·Dy_z)
               - (Dz·Dx_y·Dy_z + Dy·Dz_x·Dx_y + Dx·Dy_x·Dz_y)
        """
        # 一阶偏导 (中心差分)
        # ∂/∂z
        dphi_dz = (flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]) / 2.0
        # ∂/∂y
        dphi_dy = (flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]) / 2.0
        # ∂/∂x
        dphi_dx = (flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]) / 2.0

        # 各分量的偏导
        # dphi_dz[:, 0] = ∂φz/∂z, dphi_dz[:, 1] = ∂φy/∂z, dphi_dz[:, 2] = ∂φx/∂z
        dz_dz, dy_dz, dx_dz = dphi_dz[:, 0], dphi_dz[:, 1], dphi_dz[:, 2]
        dz_dy, dy_dy, dx_dy = dphi_dy[:, 0], dphi_dy[:, 1], dphi_dy[:, 2]
        dz_dx, dy_dx, dx_dx = dphi_dx[:, 0], dphi_dx[:, 1], dphi_dx[:, 2]

        # 变形场 φ = x + flow, Jacobian 矩阵 J = I + ∇flow
        # det(J) = det(I + ∇flow), 对角线加 1
        det = (
            (1 + dz_dz) * ((1 + dy_dy) * (1 + dx_dx) - dy_dx * dx_dy)
            - dz_dy * (dy_dz * (1 + dx_dx) - dy_dx * dx_dz)
            + dz_dx * (dy_dz * dx_dy - (1 + dy_dy) * dx_dz)
        )

        return det


# ============================================================
#  组合损失
# ============================================================

class RegistrationLoss3D(nn.Module):
    """
    3D 配准组合损失

    L = w_sim · L_sim + w_smooth · L_smooth + w_jac · L_jac

    参数
    ----
    sim_loss    : 相似性损失模块
    smooth_loss : 正则化损失模块
    sim_weight  : 相似性权重
    smooth_weight : 正则化权重
    jac_weight  : Jacobian 惩罚权重 (0 表示不使用)
    """

    def __init__(
        self,
        sim_loss: nn.Module = None,
        smooth_loss: nn.Module = None,
        sim_weight: float = 1.0,
        smooth_weight: float = 0.5,
        jac_weight: float = 0.1,
    ):
        super().__init__()
        self.sim_loss = sim_loss or NCCLoss3D(window_size=9)
        self.smooth_loss = smooth_loss or GradientLoss3D(penalty="l2")
        self.jac_loss = JacobianLoss()
        self.sim_weight = sim_weight
        self.smooth_weight = smooth_weight
        self.jac_weight = jac_weight

    def forward(
        self, fixed: torch.Tensor, warped: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        返回
        ----
        total_loss : 标量
        loss_dict  : 各项损失值 (用于日志)
        """
        l_sim = self.sim_loss(fixed, warped)
        l_smooth = self.smooth_loss(flow)
        l_jac = self.jac_loss(flow)

        total = (
            self.sim_weight * l_sim
            + self.smooth_weight * l_smooth
            + self.jac_weight * l_jac
        )

        loss_dict = {
            "sim": float(l_sim.item()),
            "smooth": float(l_smooth.item()),
            "jac": float(l_jac.item()),
            "total": float(total.item()),
        }

        return total, loss_dict


if __name__ == "__main__":
    print("测试 3D 配准损失函数...")
    fixed = torch.randn(1, 1, 32, 32, 32)
    warped = torch.randn(1, 1, 32, 32, 32)
    flow = torch.randn(1, 3, 32, 32, 32) * 0.1

    ncc = NCCLoss3D(9)
    print(f"  NCC3D:      {ncc(fixed, warped):.4f}")

    mse = MSELoss3D()
    print(f"  MSE3D:      {mse(fixed, warped):.4f}")

    lncc = LNCCLoss3D((5, 9))
    print(f"  LNCC3D:     {lncc(fixed, warped):.4f}")

    grad = GradientLoss3D("l2")
    print(f"  Gradient3D: {grad(flow):.4f}")

    bend = BendingEnergyLoss3D()
    print(f"  Bending3D:  {bend(flow):.4f}")

    jac = JacobianLoss()
    print(f"  Jacobian:   {jac(flow):.4f}")

    combo = RegistrationLoss3D()
    total, ld = combo(fixed, warped, flow)
    print(f"  组合损失:   {total:.4f}  {ld}")
