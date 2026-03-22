import torch
from PIL import Image
import torchvision.transforms as T

def padding_and_resize_pil(image: Image.Image, mode: str = 'RGB', output_w: int = 900, output_h: int = 800) -> torch.Tensor:
    if mode == 'RGB' and image.mode != 'RGB':
        image = image.convert('RGB')
    elif mode == 'thermal' and image.mode != 'L':
        image = image.convert('L')

    w, h = image.size
    target_ratio = output_w / output_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)

    if mode == 'RGB':
        padded_image = Image.new('RGB', (new_w, new_h), (0, 0, 0))
    else:
        padded_image = Image.new('L', (new_w, new_h), 0)

    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    padded_image.paste(image, (paste_x, paste_y))

    transform = T.Compose([
        T.Resize((output_h, output_w)),
        T.ToTensor()
    ])

    tensor = transform(padded_image).clone()

    return tensor


def padding_and_resize_tensor(tensor: torch.Tensor, output_h: int = 800, output_w: int = 900) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    shape_len = len(tensor.shape)
    if shape_len not in [3, 4]:
        raise ValueError(f"Unsupported tensor shape length: {shape_len}")

    if shape_len == 3:
        if tensor.shape[0] == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[2] == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Cannot find channel dimension of size 3 in shape {tensor.shape}")
    else:
        if tensor.shape[1] == 3:
            pass
        elif tensor.shape[3] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Cannot find channel dimension of size 3 in shape {tensor.shape}")

    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    b, c, h, w = tensor.shape
    target_ratio = output_w / output_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = output_w
        new_h = int(output_w / current_ratio)
        pad_h = output_h - new_h
        pad_w = 0
    else:
        new_h = output_h
        new_w = int(output_h * current_ratio)
        pad_h = 0
        pad_w = output_w - new_w

    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    tensor = torch.nn.functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=0
    )

    return tensor