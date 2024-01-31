import time
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import timm
from utils import save_image_to_gridfs
assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check
from PIL import Image, ImageDraw

from misc import make_grid
import models_mae_cross
device = torch.device('cpu')


"""
python demo.py
"""
from sklearn.cluster import DBSCAN

def apply_dbscan(points, eps=5, min_samples=1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Generate a list of cluster centers
    cluster_centers = []
    for k in range(n_clusters_):
        class_member_mask = (labels == k)
        xy = np.array(points)[class_member_mask]
        cluster_centers.append(xy.mean(axis=0).tolist())

    return cluster_centers, n_clusters_, n_noise_

class measure_time(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = (time.perf_counter_ns() - self.start) / 1e9

def find_high_density_points(density_map, threshold=2.0):
    points = []
    height, width = density_map.shape
    for y in range(height):
        for x in range(width):
            if density_map[y, x] > threshold:
                points.append((x, y))
    return points

def overlay_points_on_image(image, points, color=(0, 255, 0), radius=2):
    draw = ImageDraw.Draw(image)
    for point in points:
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
    return image

def plot_heatmap(density_map, count, buffer):
    # Make sure that this function now writes to a BytesIO buffer instead of saving to a file path
    fig, ax = plt.subplots()
    im = ax.imshow(density_map.cpu().numpy(), cmap='hot', interpolation='nearest')
    ax.text(0, 0, f'Count: {count:.2f}', color='white', fontsize=12, va='top', ha='left', backgroundcolor='black')
    plt.colorbar(im)
    fig.canvas.draw()  # This line draws the figure on the canvas so that it can be saved
    plt.savefig(buffer, format='png')
    plt.close()


def load_image(filepath):
    image = Image.open(filepath)
    image.load()
    W, H = image.size

    # Resize the image size so that the height is 384
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    scale_factor_H = float(new_H) / H
    scale_factor_W = float(new_W) / W
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    # Coordinates of the exemplar bound boxes
    # The left upper corner and the right lower corner
    bboxes = [
        [[136, 98], [173, 127]],
        [[209, 125], [242, 150]],
        [[212, 168], [258, 200]]
    ]
    boxes = list()
    rects = list()
    for bbox in bboxes:
        x1 = int(bbox[0][0] * scale_factor_W)
        y1 = int(bbox[0][1] * scale_factor_H)
        x2 = int(bbox[1][0] * scale_factor_W)
        y2 = int(bbox[1][1] * scale_factor_H)
        rects.append([y1, x1, y2, x2])
        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
        bbox = transforms.Resize((64, 64))(bbox)
        boxes.append(bbox.numpy())

    boxes = np.array(boxes)
    boxes = torch.Tensor(boxes)

    return image, boxes, rects


def run_one_image(samples, boxes, pos, model,fs):
    _, _, h, w = samples.shape

    s_cnt = 0
    for rect in pos:
        if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
            s_cnt += 1
    if s_cnt >= 1:
        r_densities = []
        r_images = []
        r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))  # 1
        r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))  # 3
        r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))  # 7
        r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))  # 2
        r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))  # 4
        r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))  # 8
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))  # 5
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))  # 6
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))  # 9

        pred_cnt = 0
        with measure_time() as et:
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1
                with torch.no_grad():
                    while start + 383 < w:
                        output, = model(r_image[:, :, :, start:start + 384], boxes, 3)
                        output = output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                        d1 = b1(output[:, 0:prev - start + 1])
                        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                        d2 = b2(output[:, prev - start + 1:384])

                        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                        density_map_l = b3(density_map[:, 0:start])
                        density_map_m = b1(density_map[:, start:prev + 1])
                        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                        density_map_r = b4(density_map[:, prev + 1:w])

                        density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                        prev = start + 383
                        start = start + 128
                        if start + 383 >= w:
                            if start == w - 384 + 128:
                                break
                            else:
                                start = w - 384

                pred_cnt += torch.sum(density_map / 60).item()
                r_densities += [density_map]
    else:
        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with measure_time() as et:
            with torch.no_grad():
                while start + 383 < w:
                    output, = model(samples[:, :, :, start:start + 384], boxes, 3)
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()


    e_cnt = 0
    for rect in pos:
        e_cnt += torch.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60).item()
    e_cnt = e_cnt / 3
    if e_cnt > 1.8:
        pred_cnt /= e_cnt

    # Visualize the prediction
    fig = samples[0]
    box_map = torch.zeros([fig.shape[1], fig.shape[2]])
    box_map = box_map.to(device, non_blocking=True)
    for rect in pos:
        for i in range(rect[2] - rect[0]):
            box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[1], fig.shape[2] - 1)] = 10
            box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[3], fig.shape[2] - 1)] = 10
        for i in range(rect[3] - rect[1]):
            box_map[min(rect[0], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
            box_map[min(rect[2], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
    box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
    pred = density_map.unsqueeze(0).repeat(3, 1, 1) if s_cnt < 1 \
        else make_grid(r_densities, h, w).unsqueeze(0).repeat(3, 1, 1)
    fig = fig + box_map + pred / 2
    fig = torch.clamp(fig, 0, 1)
   # Generate overlaid image with high-density points
    
    high_density_points = find_high_density_points(density_map.cpu().numpy(), threshold=1.0)
    pred_cnt = len(high_density_points)  # Count is now the number of high-density points
    original_image = TF.to_pil_image(samples.cpu().squeeze(0))
    overlaid_image = overlay_points_on_image(original_image, high_density_points)

    # Save or display the overlaid image
    overlaid_image_buffer = BytesIO()
    overlaid_image.save(overlaid_image_buffer, format='png')
    overlaid_image_buffer.seek(0)
    overlaid_image_file_id = fs.put(overlaid_image_buffer, filename='overlaid_image.png', content_type='image/png')

    return pred_cnt, et.duration, str(overlaid_image_file_id)


# Prepare model
model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
model.to(device)
model_without_ddp = model

checkpoint = torch.load('./checkpoint-400.pth', map_location='cpu')
model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
print("Resume checkpoint %s" % './checkpoint-400.pth')

model.eval()

def run_demo(filepath, fs):
    samples, boxes, pos = load_image(filepath)
    samples = samples.unsqueeze(0).to(device, non_blocking=True)
    boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

    result, elapsed_time, heatmap_file_id = run_one_image(samples, boxes, pos, model, fs)

    return result, elapsed_time, heatmap_file_id