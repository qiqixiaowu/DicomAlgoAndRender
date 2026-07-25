"""验证 Jacobian 行列式修复"""
import torch, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from utils.metrics3d import jacobian_determinant_3d, jacobian_stats_3d, folding_ratio_3d

# Test 1: zero flow → det(I) = 1 everywhere, folding = 0
flow_zero = torch.zeros(1, 3, 16, 16, 16)
stats = jacobian_stats_3d(flow_zero)
print("Zero flow:", {k: f"{v:.6f}" for k, v in stats.items()})
assert abs(stats["mean"] - 1.0) < 1e-5, f"Expected mean=1.0, got {stats['mean']}"
assert stats["folding_ratio"] < 1e-5, f"Expected folding=0, got {stats['folding_ratio']}"
print("  PASS: zero flow -> det~1, folding~0")

# Test 2: small smooth flow → det near 1, folding near 0
torch.manual_seed(42)
flow_small = torch.randn(1, 3, 16, 16, 16) * 0.01
stats2 = jacobian_stats_3d(flow_small)
print("Small flow:", {k: f"{v:.6f}" for k, v in stats2.items()})
assert stats2["folding_ratio"] < 0.1, f"Expected low folding, got {stats2['folding_ratio']}"
print("  PASS: small flow -> low folding")

# Test 3: large random flow → some folding expected
flow_large = torch.randn(1, 3, 16, 16, 16) * 0.5
stats3 = jacobian_stats_3d(flow_large)
print("Large flow:", {k: f"{v:.6f}" for k, v in stats3.items()})
print("  (folding > 0 expected for large random flow)")

print("\n✓ All Jacobian tests passed!")
