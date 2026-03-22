import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import cv2
from PIL import Image
import torch.nn.functional as F

def IoU_xywh(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1.unbind(-1)
    x2, y2, w2, h2 = bbox2.unbind(-1)

    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2

    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2

    intersect_min_x = torch.max(x1_min, x2_min)
    intersect_min_y = torch.max(y1_min, y2_min)
    intersect_max_x = torch.min(x1_max, x2_max)
    intersect_max_y = torch.min(y1_max, y2_max)

    intersect_w = torch.clamp(intersect_max_x - intersect_min_x, min=0)
    intersect_h = torch.clamp(intersect_max_y - intersect_min_y, min=0)

    intersect_area = intersect_w * intersect_h
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersect_area

    iou = intersect_area / union_area.clamp(min=1e-6)
    return iou.unsqueeze(-1)

def save_patch(rgb, thermal, path):
    rgb = (rgb * 255).type(torch.uint8).numpy()
    thermal = (thermal * 255).type(torch.uint8).numpy()
    Image.fromarray(rgb).save(path + '_rgb.png')
    Image.fromarray(thermal[:,:,0]).save(path + '_thermal.png')

def tis(image, save_path, mode='RGB'):
    """
    Converts an image to a torch.Tensor, checks for normalization, rearranges dimensions if necessary,
    and saves the image.

    Args:
    image: An image that can be converted to a tensor. It can be a 3D or 4D array.
    save_path: Path where the image will be saved.
    mode: Color mode of the image ('RGB' or 'BGR').
    """
    if '/' in save_path:
        save_dir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if isinstance(image, torch.Tensor):
        image = image.clone().cpu()

    if image.max() > 1:
        image = image / 255.0

    if image.dim() == 4:
        if image.shape[0] != 1:
            raise ValueError("Dimension 4D but shape[0] not 1")
        image = image.squeeze(0)
        if image.shape[0] == 3:
            pass
        elif image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError("Invalid image shape, expected 3 channels")
    elif image.dim() == 3:
        if image.shape[0] == 3:
            pass
        elif image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError("Invalid image shape, expected 3 channels")
    else:
        raise ValueError("Invalid image dimensions, expected 3D or 4D tensor")

    if mode.upper() == 'BGR':
        image = image[[2, 1, 0], :, :]

    image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(image)
    image.save(save_path)

def gtis(image, save_path):
    """
    Converts an image to a torch.Tensor, checks for normalization, rearranges dimensions if necessary,
    and saves the image as a grayscale image.

    Args:
    image: An image that can be converted to a tensor. It can be a 2D, 3D, or 4D array.
    save_path: Path where the image will be saved.
    """

    if '/' in save_path:
        save_dir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if isinstance(image, torch.Tensor):
        image = image.clone().cpu()

    if image.max() > 1:
        image = image / 255.0

    if image.dim() == 4:
        if image.shape[0] != 1:
            raise ValueError("Dimension 4D but shape[0] not 1")
        image = image.squeeze(0)
        if image.shape[0] == 1:
            pass
        elif image.shape[2] == 1:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError("Invalid image shape, expected 1 channel")
    elif image.dim() == 3:
        if image.shape[0] == 1:
            pass
        elif image.shape[2] == 1:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError("Invalid image shape, expected 1 channel")
    elif image.dim() == 2:
        image = image.unsqueeze(0)
    else:
        raise ValueError("Invalid image dimensions, expected 2D, 3D or 4D tensor")

    image = (image.clamp(0, 1) * 255).byte().squeeze(0).numpy()
    image = Image.fromarray(image, mode='L')
    image.save(save_path)

def batch_to_grayscale(batch):
    batch_size = batch.shape[0]
    grayscale_batch = torch.zeros_like(batch)
    for i in range(batch_size):
        img = batch[i]
        img = (img.numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)), 'RGB')
        img_gray_pil = img_pil.convert('L')
        img_gray_rgb_pil = Image.merge('RGB', (img_gray_pil, img_gray_pil, img_gray_pil))
        img_gray_rgb = np.asarray(img_gray_rgb_pil).transpose(2, 0, 1) / 255.0
        grayscale_batch[i] = torch.tensor(img_gray_rgb, dtype=batch.dtype)
    return grayscale_batch

def save_visualize_boxes(rgb, thermal, boxes, scores, rgb_path, thermal_path):
    '''
    rgb: tensor, [3, H, W], range [0, 1]
    thermal: tensor, [1, H, W], range [0, 1]
    boxes: tensor, [N, 4], xyxy, not normalized
    scores: tensor, [N], range [0, 1]
    '''

    rgb = rgb.detach()
    thermal = thermal.detach()
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    thermal_np = (thermal.squeeze().cpu().numpy() * 255).astype(np.uint8)
    thermal_np = np.stack([thermal_np] * 3, axis=-1)

    color = (0, 255, 0)
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    if boxes is not None and scores is not None and len(boxes) > 0 and len(scores) > 0:
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.int().tolist()
            score = score.item()
            cv2.rectangle(rgb_np, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(thermal_np, (x1, y1), (x2, y2), color, thickness)

            label = f"{score:.3f}"
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y = max(y1 - 10, label_size[1])
            cv2.putText(rgb_np, label, (x1, label_y), font, font_scale, color, font_thickness)
            cv2.putText(thermal_np, label, (x1, label_y), font, font_scale, color, font_thickness)

    cv2.imwrite(rgb_path, rgb_np)
    cv2.imwrite(thermal_path, thermal_np)

def write_det_result(det_result, img_tensor, box_config = {'conf_thr': 0.0, 'max_num': 1}):
    if det_result is None:
        return img_tensor
    elif det_result[1][0].numel() == 0:
        return img_tensor
    else:
        img_np = (img_tensor[0].permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
        img_np = img_np.copy()

        boxes = det_result[0][0]
        scores = det_result[1][0]
        for i in range(len(boxes)):
            if scores[i] > box_config['conf_thr'] and i < box_config['max_num']:
                x1, y1, x2, y2 = boxes[i].int().tolist()
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)

                conf_text = f"{scores[i]*100:.2f}%"
                cv2.putText(img_np, conf_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0,
                           (0, 255, 0),
                           2)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    return img_tensor

def get_material_patch(img_path, h, w, grid_size=20, kernel_size=15, sigma=5):
    img = Image.open(img_path)
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

    grid_h = h // grid_size + (1 if h % grid_size else 0)
    grid_w = w // grid_size + (1 if w % grid_size else 0)

    offset_h = torch.randint(0, 10000, (grid_h, grid_w))
    offset_w = torch.randint(0, 10000, (grid_h, grid_w))

    offset_h = F.interpolate(
        offset_h.unsqueeze(0).unsqueeze(0).float(),
        size=(h, w),
        mode='nearest'
    ).squeeze(0).squeeze(0)

    offset_w = F.interpolate(
        offset_w.unsqueeze(0).unsqueeze(0).float(),
        size=(h, w),
        mode='nearest'
    ).squeeze(0).squeeze(0)

    sample_h = (offset_h % img_tensor.shape[1]).float() / (img_tensor.shape[1] - 1)
    sample_w = (offset_w % img_tensor.shape[2]).float() / (img_tensor.shape[2] - 1)

    grid = torch.stack([sample_w * 2 - 1, sample_h * 2 - 1], dim=-1).unsqueeze(0)

    patch = F.grid_sample(
        img_tensor.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0)

    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_h = kernel_1d.view(1, 1, kernel_size, 1)
    kernel_w = kernel_1d.view(1, 1, 1, kernel_size)

    patch = patch.unsqueeze(0)
    result = []

    for i in range(3):
        channel = patch[:,i:i+1,:,:]
        temp = F.pad(channel, (kernel_size//2, kernel_size//2, 0, 0), mode='reflect')
        temp = F.conv2d(temp, kernel_w)
        temp = F.pad(temp, (0, 0, kernel_size//2, kernel_size//2), mode='reflect')
        temp = F.conv2d(temp, kernel_h)
        result.append(temp)

    patch = torch.cat(result, dim=1)

    return patch.squeeze(0)

def random_hsv_augmentation(img_tensor):
    img = img_tensor.clone()
    if img.dim() == 3:
        img = img.unsqueeze(0)

    eps = 1e-6
    r, g, b = img[:, 0, ...], img[:, 1, ...], img[:, 2, ...]

    temp = 10.0
    rgb_stack = torch.stack([r, g, b], dim=1)
    max_weights = torch.softmax(rgb_stack * temp, dim=1)
    min_weights = torch.softmax(-rgb_stack * temp, dim=1)

    max_rgb = (rgb_stack * max_weights).sum(dim=1)
    min_rgb = (rgb_stack * min_weights).sum(dim=1)
    diff = max_rgb - min_rgb + eps

    h = torch.zeros_like(max_rgb)

    is_r_max = torch.exp(-5 * torch.abs(r - max_rgb))
    is_g_max = torch.exp(-5 * torch.abs(g - max_rgb))
    is_b_max = torch.exp(-5 * torch.abs(b - max_rgb))
    total_weights = is_r_max + is_g_max + is_b_max + eps

    h += is_r_max / total_weights * ((60 * (g - b) / diff + 360) % 360)
    h += is_g_max / total_weights * ((60 * (b - r) / diff + 120))
    h += is_b_max / total_weights * ((60 * (r - g) / diff + 240))

    s = diff / (max_rgb + eps)
    v = max_rgb

    device = img.device
    batch_size = img.shape[0]
    h_shift = torch.tanh(torch.randn(batch_size, 1, 1, device=device)) * 3
    s_shift = torch.tanh(torch.randn(batch_size, 1, 1, device=device)) * 0.03
    v_shift = torch.tanh(torch.randn(batch_size, 1, 1, device=device)) * 0.03

    h = (h + h_shift) % 360
    s = torch.clamp(s + s_shift, 0, 1)
    v = torch.clamp(v + v_shift, 0, 1)

    h_i = (h / 60).long() % 6
    f = (h / 60) - h_i.float()

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = torch.zeros_like(img)

    for i in range(6):
        mask = (h_i == i)
        if not mask.any():
            continue

        if i == 0:
            rgb[:, 0, ...][mask] = v[mask]
            rgb[:, 1, ...][mask] = t[mask]
            rgb[:, 2, ...][mask] = p[mask]
        elif i == 1:
            rgb[:, 0, ...][mask] = q[mask]
            rgb[:, 1, ...][mask] = v[mask]
            rgb[:, 2, ...][mask] = p[mask]
        elif i == 2:
            rgb[:, 0, ...][mask] = p[mask]
            rgb[:, 1, ...][mask] = v[mask]
            rgb[:, 2, ...][mask] = t[mask]
        elif i == 3:
            rgb[:, 0, ...][mask] = p[mask]
            rgb[:, 1, ...][mask] = q[mask]
            rgb[:, 2, ...][mask] = v[mask]
        elif i == 4:
            rgb[:, 0, ...][mask] = t[mask]
            rgb[:, 1, ...][mask] = p[mask]
            rgb[:, 2, ...][mask] = v[mask]
        elif i == 5:
            rgb[:, 0, ...][mask] = v[mask]
            rgb[:, 1, ...][mask] = p[mask]
            rgb[:, 2, ...][mask] = q[mask]

    alpha = 0.8
    rgb = alpha * img + (1 - alpha) * rgb

    if img_tensor.dim() == 3:
        rgb = rgb.squeeze(0)

    return rgb

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = "assets/material/grey.png"
    grid_size = 1
    kernel_size = 1
    sigma = 5

    plt.figure(figsize=(15, 6))

    for i in range(3):
        patch = get_material_patch(
            img_path,
            340,
            800,
            grid_size=grid_size,
            kernel_size=kernel_size,
            sigma=sigma
        )
        patch = patch.permute(1, 2, 0).numpy()

        plt.subplot(1, 3, i+1)
        plt.imshow(patch)
        plt.axis('off')
        plt.title(f'Random Patch {i+1}')

    plt.tight_layout()
    plt.savefig('random_patches.png')
