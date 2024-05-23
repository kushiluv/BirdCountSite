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
import matplotlib
matplotlib.use('Agg')
import timm
from model_files.utils import save_image_to_gridfs
assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check
from sklearn.cluster import DBSCAN
import numpy as np
from model_files.misc import make_grid
import model_files.models_mae_cross as models_mae_cross
import matplotlib.cm as cm
device = torch.device('cuda')
import warnings  
global model, model_without_ddp
warnings.filterwarnings('ignore')

"""
python demo.py
"""


class measure_time(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = (time.perf_counter_ns() - self.start) / 1e9


def plot_heatmap(density_map, count, buffer):
    # Make sure that this function now writes to a BytesIO buffer instead of saving to a file path
    fig, ax = plt.subplots()
    im = ax.imshow(density_map.cpu().numpy(), cmap='viridis')
    # ax.text(0, 0, f'Count: {count:.2f}', color='white', fontsize=12, va='top', ha='left', backgroundcolor='black')
    # plt.colorbar(im)
    fig.canvas.draw()  # This line draws the figure on the canvas so that it can be saved
    plt.savefig(buffer, format='png')
    plt.close()

def load_model(checkpoint_path):
    global model, model_without_ddp
    model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model_without_ddp = model
    print("Loaded the new model" , checkpoint_path)
    return model

def load_image(file_id,fs):
    # Open the image file
    grid_out = fs.get(file_id)
    # Use BytesIO to create a file-like object from the binary data
    image_data = BytesIO(grid_out.read())
    image_data.seek(0)  # Important: seek to the beginning of the file-like object
    image = Image.open(image_data).convert('RGB')
    W, H = image.size

    # Resize the image size so that the height is 384
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    scale_factor_H = float(new_H) / H
    scale_factor_W = float(new_W) / W
    print(f"Original image size: {W}x{H}")
    print(f"Processed image size: {new_W}x{new_H}")
    print(f"Scaling factors: {scale_factor_W}, {scale_factor_H}")
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    # Coordinates of the exemplar bound boxes
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


def run_one_image(samples, boxes, pos, model,fs, orig_image_size):
    _, _, h, w = samples.shape
    orig_h, orig_w = orig_image_size

    # Calculate the scaling factors
    print("test : ",orig_h, orig_w, h, w)
    scale_factor_H = orig_h / h
    scale_factor_W = orig_w / w
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

    # Normalize density_map for visualization
    density_map = density_map.to('cuda')
    samples = samples.to('cuda')

    # Normalize density_map for visualization
    density_normalized = density_map / density_map.max()

    # Convert density map to RGBA image using a colormap
    colormap = cm.get_cmap('jet')  # You can choose any available colormap
    density_colormap = colormap(density_normalized.cpu().detach().numpy())  # Move to CPU and convert to numpy

    # Convert RGBA image to RGB by ignoring the alpha channel
    density_rgb = density_colormap[...,:3]

    # Convert to tensor and move to GPU
    density_rgb_tensor = torch.from_numpy(density_rgb).float().permute(2, 0, 1).to('cuda')

    # Resize density_rgb_tensor to match the size of the original image
    density_resized = TF.resize(density_rgb_tensor, samples.shape[2:])

    # Blend the original image with the density map
    # Adjust alpha to change the transparency of the overlay
    alpha = 0.5
    blended_image = (1 - alpha) * samples[0] + alpha * density_resized
    
    # Clamp the values to be between 0 and 1
    blended_image_clamped = torch.clamp(blended_image, 0, 1)
    
    # Save or display the blended image
    # Convert blended_image_clamped to PIL image to save or display
    blended_image_pil = TF.to_pil_image(blended_image_clamped.cpu())
    blended_image_buffer = BytesIO()
    blended_image_pil.save(blended_image_buffer, format='PNG')
    blended_image_buffer.seek(0)
    blended_image_file_id = fs.put(blended_image_buffer, filename='blended_heatmap.png', content_type='image/png')
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
    heatmap_buffer = BytesIO()
    plot_heatmap(density_map, pred_cnt, heatmap_buffer)
    heatmap_buffer.seek(0)
    heatmap_file_id = fs.put(heatmap_buffer, filename='heatmap.png', content_type='image/png')
    pred_cnt = int(pred_cnt + 0.99)
    # Thresholding the density map
   
    # Convert cluster centers to a format suitable for JSON serialization
    # This will be useful for sending data to the frontend
    
     # Include cluster_centers_json in the return statement
    return pred_cnt, et.duration, str(blended_image_file_id), density_map
    
    

def compute_clusters_for_range(density_map, scale_factors):
    cluster_centers_sets = []
    # Define sets of parameters for thresholds, eps, and min_samples
    parameters = [
        {'threshold': 0.999, 'eps': 0.5, 'min_samples': 1},
        {'threshold': 0.9, 'eps': 1, 'min_samples': 2},
        {'threshold': 0.8, 'eps': 2, 'min_samples': 2},
        {'threshold': 0.7, 'eps': 3, 'min_samples': 2},
        {'threshold': 0.6, 'eps': 4, 'min_samples': 2},
        {'threshold': 0.5, 'eps': 5, 'min_samples': 2},
        {'threshold': 0.4, 'eps': 6, 'min_samples': 2},
        {'threshold': 0.3, 'eps': 7, 'min_samples': 2},
        {'threshold': 0.2, 'eps': 8, 'min_samples': 2},
        {'threshold': 0.1, 'eps': 9, 'min_samples': 2},
    ]
    
    for param in parameters:
        binary_mask = threshold_density_map(density_map, param['threshold'])
        cluster_centers = cluster_points(binary_mask, eps=param['eps'], min_samples=param['min_samples'])
        # Adjust cluster centers based on scale_factors if necessary
        adjusted_centers = [{'x': int(center[0] * scale_factors['W']), 'y': int(center[1] * scale_factors['H'])} for center in cluster_centers]
        cluster_centers_sets.append(adjusted_centers)
    
    return cluster_centers_sets

def threshold_density_map(density_map, threshold):
    # Apply threshold
    binary_mask = density_map > threshold
    return binary_mask

def cluster_points(binary_mask, eps, min_samples):
    # Ensure binary_mask is on CPU before using numpy functions
    binary_mask = binary_mask.cpu().numpy()

    # Find coordinates of potential locations
    y, x = np.where(binary_mask)
    points = np.array(list(zip(x, y)))
    print(points)
    print(x, y)

    # Check if points array is empty
    if points.size == 0:
        print("No points to cluster")
        return []  # or handle this case as appropriate

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    cluster_centers = []
    for label in np.unique(clustering.labels_):
        if label != -1:  # Ignore noise points
            members = points[clustering.labels_ == label]
            center = members.mean(axis=0)
            cluster_centers.append(center)
    
    return cluster_centers


# Prepare model
model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
model.to(device)
model_without_ddp = model

checkpoint = torch.load('model_files/pth/original.pth', map_location='cuda')
model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
# print("Resume checkpoint %s" % './checkpoint-400.pth')

model.eval()

def run_demo(file_id, fs ,checkpoint_path=None):
    if not checkpoint_path:
        checkpoint_path = './default_checkpoint.pth'  # default path
    samples, boxes, pos = load_image(file_id, fs)
    samples = samples.unsqueeze(0).to(device, non_blocking=True)
    boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
    orig_image_size = samples.shape[2:]  # Capture the original image size

    # Now, run_one_image returns the density_map as well
    pred_cnt, elapsed_time, heatmap_file_id, density_map = run_one_image(samples, boxes, pos, model, fs, orig_image_size)

    # Compute scale factors based on the original image size and the processed size
    scale_factors = {'W': orig_image_size[1]/density_map.shape[1], 'H': orig_image_size[0]/density_map.shape[0]}

    # Generate multiple sets of cluster centers
    cluster_centers_sets = compute_clusters_for_range(density_map, scale_factors)

    return pred_cnt, elapsed_time, heatmap_file_id, cluster_centers_sets, orig_image_size