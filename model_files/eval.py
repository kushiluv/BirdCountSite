from itertools import chain
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import timm
import numpy as np
from PIL import ImageDraw
import os
import time
import model_files.models_mae_cross as models_mae_cross
from util.misc import measure_time
import argparse

assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shot_num = 0


def load_image(img_path: str):
    image = Image.open(img_path).convert('RGB')
    image.load()
    W, H = image.size

    # Resize the image size so that the height is 384
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    # Coordinates of the exemplar bound boxes
    # Not needed for zero-shot counting
    boxes = torch.Tensor([])
    return image, boxes, W, H


def run_one_image(samples, boxes, model,  img_name, old_w, old_h):
    _, _, h, w = samples.shape

    density_map = torch.zeros([h, w])
    density_map = density_map.to(device, non_blocking=True)
    start = 0
    prev = -1
    with measure_time() as et:
        with torch.no_grad():
            while start + 383 < w:
                output, = model(samples[:, :, :, start:start + 384], boxes, shot_num)
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

    # Visualize the prediction
    fig = samples[0]
    pred_fig = torch.stack((density_map, torch.zeros_like(density_map), torch.zeros_like(density_map)))
    count_im = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(count_im)
    draw.text((w-70, h-50), f"{pred_cnt:.3f}", (255, 255, 255))
    count_im = np.array(count_im).transpose((2, 0, 1))
    count_im = torch.tensor(count_im, device=device)
    fig = fig / 2 + pred_fig / 2 + count_im
    fig = torch.clamp(fig, 0, 1)
    fig = transforms.Resize((old_h, old_w))(fig)
    # torchvision.utils.save_image(fig, output_path / f'viz_{img_name}.jpg')
    return pred_cnt, et


def mae_evaluation(model_path):

    images_dir = "model_files/eval_db/Images"
    gt_dir = "model_files/eval_db/GT"

    image_filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    args = argparse.Namespace()
    args.input_path = ""
    args.model_path = model_path    

    # Prepare model
    model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
    model.to(device)
    model_without_ddp = model

    checkpoint = torch.load(args.model_path, map_location='cuda')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print(f"Resume checkpoint {args.model_path}")

    model.eval()

    Loss = 0
    Count = 0
    start_time = time.time()

    for image_name in tqdm(image_filenames, desc="Processing Images"):
        image_path = os.path.join(images_dir, image_name + '.jpg')
        gt_path = os.path.join(gt_dir, image_name + '.npy')
        args.input_path = Path(image_path)
        samples, boxes, old_w, old_h = load_image(args.input_path)
        samples = samples.unsqueeze(0).to(device, non_blocking=True)
        boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
        result, elapsed_time = run_one_image(samples, boxes, model, args.input_path.stem, old_w, old_h)
        Prediction = result
        
        # Counting error
        gt = np.load(gt_path)
        gt = gt.sum()
        Count += 1
        Loss += abs(Prediction - gt) / gt

    total_time = time.time() - start_time
    print("MAE:", Loss / Count)
    print(f"Total elapsed time: {total_time:.2f} seconds")
    return Loss/Count
    # print(f"Average time per image: {total_time / Count:.2f} seconds")