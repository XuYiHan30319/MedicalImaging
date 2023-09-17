import torch

window_size = [5, 10]
coords_h = torch.arange(5)
coords_w = torch.arange(10)
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2,50
# print(coords_flat)
# 2,50,1 - 2,1,50
# 2,50,50
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

relative_coords = relative_coords.permute(1, 2, 0).contiguous()
relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += window_size[1] - 1

relative_coords[:, :, 0] *= 2 * window_size[1] - 1
print(relative_coords)
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
