#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import librosa
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from options.test_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_magphase
from utils import utils
import soundfile as sf
import matplotlib.pyplot as plt

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def get_separated_audio(outputs, batch_data, opt):
    mag_mix = batch_data['audio_mix_mags']
    phase_mix = batch_data['audio_mix_phases']
    pred_masks_ = outputs['pred_mask']
    B = mag_mix.size(0)
    
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, pred_masks_.size(3), warp=False)).to(opt.device)
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
        
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    pred_mag = mag_mix[0, 0] * pred_masks_linear[0, 0]
    preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=opt.stft_hop, length=opt.audio_window)
    return preds_wav

def get_detection_at_time(sliding_window_start, audio_sampling_rate, detectionResult, frame_PATH, vision_transform):
    current_time_sec = sliding_window_start / audio_sampling_rate
    estimated_frame_id = int(current_time_sec * 8) + 1  
    
    idx = (np.abs(detectionResult[:, 0] - estimated_frame_id)).argmin()
    det_box = detectionResult[idx, :]
    
    frame_path = os.path.join(frame_PATH, "%06d.png" % det_box[0])

    if not os.path.exists(frame_path):
        det_box = detectionResult[np.argmax(detectionResult[:,2]), :] 
        frame_path = os.path.join(frame_PATH, "%06d.png" % det_box[0])
        
    img = Image.open(frame_path).convert('RGB').crop((det_box[-4], det_box[-3], det_box[-2], det_box[-1]))
    return vision_transform(img).unsqueeze(0)

def main():
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")

    builder = ModelBuilder()
    opt.with_fc = True if opt.identity_feature_dim != 512 else False

    net_visual = builder.build_visual(pool_type=opt.visual_pool, weights=opt.weights_visual)
    net_unet = builder.build_unet(
        unet_num_layers=opt.unet_num_layers, ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc, output_nc=opt.unet_output_nc, weights=opt.weights_unet)
    
    if opt.with_additional_scene_image: opt.number_of_classes += 1
    
    net_classifier = builder.build_classifier(
        pool_type=opt.classifier_pool, num_of_classes=opt.number_of_classes,
        input_channel=opt.unet_output_nc, weights=opt.weights_classifier)
        
    nets = (net_visual, net_unet, net_classifier, net_classifier, net_classifier)

    model = AudioVisualModel(nets, opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)
    model.eval()

    test_file = "/media/data4/home/lanlt/ducanh/dataset_av/hdf5_split/test.txt"
    with open(test_file, 'r') as f:
        all_videos = [line.strip() for line in f.readlines() if line.strip()]

    import random
    random.seed(42) 
    random.shuffle(all_videos)
    test_pairs = [(all_videos[i], all_videos[i+1]) for i in range(0, len(all_videos)-1, 2)]

    for i, (video_name1, video_name2) in enumerate(test_pairs):
        print(f"[{i+1}/{len(test_pairs)}] Dang xu ly: {video_name1} VS {video_name2}")
        opt.video1_name = video_name1.replace(".npy", "")
        opt.video2_name = video_name2.replace(".npy", "")
        
        audio1_path = opt.video1_name.replace("yolo_top_detections", "reshape_11025") + ".wav"
        audio2_path = opt.video2_name.replace("yolo_top_detections", "reshape_11025") + ".wav"
        audio1, _ = librosa.load(audio1_path, sr=opt.audio_sampling_rate)
        audio2, _ = librosa.load(audio2_path, sr=opt.audio_sampling_rate)
            
        audio_length = min(len(audio1), len(audio2))
        audio1 = clip_audio(audio1[:audio_length])
        audio2 = clip_audio(audio2[:audio_length])
        audio_mix = (audio1 + audio2) / 2.0
    
        vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        if opt.subtract_mean:
            vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        vision_transform = transforms.Compose(vision_transform_list)
    
        detectionResult1 = np.load(video_name1)
        detectionResult2 = np.load(video_name2)
        
        frame1_PATH = opt.video1_name.replace("yolo_top_detections", "frame")
        frame2_PATH = opt.video2_name.replace("yolo_top_detections", "frame")
    
        avged_sep_audio1 = np.zeros((audio_length))
        avged_sep_audio2 = np.zeros((audio_length))
        overlap_count = np.zeros((audio_length))
        sep_audio1 = np.zeros((audio_length))
        sep_audio2 = np.zeros((audio_length))
        
        sliding_window_start = 0
        samples_per_window = opt.audio_window
        
        with torch.no_grad():
            while sliding_window_start + samples_per_window < audio_length:
                sliding_window_end = sliding_window_start + samples_per_window
                audio_segment = audio_mix[sliding_window_start:sliding_window_end]
                audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
                
                data = {
                    'audio_mix_mags': torch.FloatTensor(audio_mix_mags).unsqueeze(0),
                    'audio_mix_phases': torch.FloatTensor(audio_mix_phases).unsqueeze(0),
                    'real_audio_mags': torch.FloatTensor(audio_mix_mags).unsqueeze(0),
                    'audio_mags': torch.FloatTensor(audio_mix_mags).unsqueeze(0),
                    'labels': torch.FloatTensor(np.ones((1,1))),
                    'vids': torch.FloatTensor(np.ones((1,1)))
                }
    
                data['visuals'] = get_detection_at_time(sliding_window_start, opt.audio_sampling_rate, detectionResult1, frame1_PATH, vision_transform)
                outputs1 = model.forward(data)
                reconstructed_signal1 = get_separated_audio(outputs1, data, opt)
                sep_audio1[sliding_window_start:sliding_window_end] += reconstructed_signal1
    
                data['visuals'] = get_detection_at_time(sliding_window_start, opt.audio_sampling_rate, detectionResult2, frame2_PATH, vision_transform)
                outputs2 = model.forward(data)
                reconstructed_signal2 = get_separated_audio(outputs2, data, opt)
                sep_audio2[sliding_window_start:sliding_window_end] += reconstructed_signal2
                
                overlap_count[sliding_window_start:sliding_window_end] += 1
                sliding_window_start += int(opt.hop_size * opt.audio_sampling_rate)

        overlap_count[overlap_count == 0] = 1 
        separation1 = clip_audio(np.divide(sep_audio1, overlap_count) * 2)
        separation2 = clip_audio(np.divide(sep_audio2, overlap_count) * 2)
    
        # --- LUU OUTPUT ---
        name1 = os.path.basename(opt.video1_name)
        name2 = os.path.basename(opt.video2_name)
        output_dir = os.path.join(opt.output_dir_root, f"{name1}_VS_{name2}")
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        
        sf.write(os.path.join(output_dir, 'audio1.wav'), audio1, opt.audio_sampling_rate)
        sf.write(os.path.join(output_dir, 'audio2.wav'), audio2, opt.audio_sampling_rate)
        sf.write(os.path.join(output_dir, 'audio_mixed.wav'), audio_mix, opt.audio_sampling_rate)
        sf.write(os.path.join(output_dir, 'audio1_separated.wav'), separation1, opt.audio_sampling_rate)
        sf.write(os.path.join(output_dir, 'audio2_separated.wav'), separation2, opt.audio_sampling_rate)

if __name__ == '__main__':
    main()