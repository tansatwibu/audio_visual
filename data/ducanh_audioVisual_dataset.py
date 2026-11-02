import os
import random
import glob
import math
import h5py
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from random import randrange
from PIL import Image, ImageEnhance
from data.base_dataset import BaseDataset

# ==========================================
# AUDIO & IMAGE PROCESSING FUNCTIONS
# ==========================================
def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def augment_audio(audio):
    # Scale audio randomly between 0.5 and 1.5
    audio = audio * (random.random() + 0.5) 
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def sample_audio(audio, window):
    # Repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    return audio_sample

def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

# ==========================================
# PATH & CLASS HELPER FUNCTIONS
# ==========================================
def get_clip_name(npy_path):
    # Get original file name without extension
    # Ex: ".../dan_bau_video_10_clip_001.npy" -> "dan_bau_video_10_clip_001"
    return os.path.basename(npy_path).replace(".npy", "")

def get_vid_name(npy_path):
    # Get standard video ID to avoid mixing clips from the same video
    # Ex: "dan_bau_video_10_clip_001" -> "dan_bau_video_10"
    clip_name = get_clip_name(npy_path)
    if "_clip_" in clip_name:
        return clip_name.split("_clip_")[0]
    # Fallback to old code logic if dataset format changes
    return clip_name[0:11] 

def get_class_name(npy_path):
    # Trich xuat class name tu ten file hoac duong dan
    basename = os.path.basename(npy_path).replace(".npy", "")
    if "_video_" in basename:
        return basename.split("_video_")[0].lower()
    
    # Kiem tra danh sach cac nhac cu truyen thong
    known_classes = [
        'dan_da', 'dan_nguyet', 'dan_tranh', 'tam_thap_luc', 
        'dan_bau', 'dan_day', 'dan_nhi', 'dantrung', 
        'kenbau', 'sao_truc', 'tyba'
    ]
    path_lower = npy_path.lower()
    for cls in known_classes:
        if cls in path_lower:
            return cls
            
    # Thu lay tu thu muc cha neu folder la ten class
    parent_dir = os.path.basename(os.path.dirname(npy_path)).lower()
    if parent_dir not in ['top_detections_two_classes', 'music_solo', 'yolo_top_detections', 'reshape_11025', 'frame', '']:
        return parent_dir

    # Fallback ve video id neu khong xac dinh duoc
    return get_vid_name(npy_path)

def get_audio_path(npy_path):
    # Replace yolo_top_detections folder with reshape_11025
    audio_dir = os.path.dirname(npy_path).replace("yolo_top_detections", "reshape_11025")
    clip_name = get_clip_name(npy_path)
    return os.path.join(audio_dir, f"{clip_name}.wav")

def get_frame_path(npy_path, frame_id):
    # Replace yolo_top_detections folder with frame
    frame_dir = os.path.dirname(npy_path).replace("yolo_top_detections", "frame")
    clip_name = get_clip_name(npy_path)
    # Format frame id to 6 digits, Ex: 1 -> 000001.png
    return os.path.join(frame_dir, clip_name, f"{str(int(frame_id)).zfill(6)}.png")

def sample_object_detections(detection_bbs):
    # Get the indexes of the detections for each class
    class_index_clusters = {}
    for i in range(detection_bbs.shape[0]):
        if int(detection_bbs[i,1]) in class_index_clusters:
            class_index_clusters[int(detection_bbs[i,1])].append(i)
        else:
            class_index_clusters[int(detection_bbs[i,1])] = [i]
            
    detection2return = np.array([])
    for cls in class_index_clusters.keys():
        sampledIndex = random.choice(class_index_clusters[cls])
        if detection2return.shape[0] == 0:
            detection2return = np.expand_dims(detection_bbs[sampledIndex,:], axis=0)
        else:
            detection2return = np.concatenate((detection2return, np.expand_dims(detection_bbs[sampledIndex,:], axis=0)), axis=0)
    return detection2return

# ==========================================
# MAIN DATASET CLASS
# ==========================================
class AudioVisualMUSICDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.NUM_PER_MIX = opt.num_per_mix
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)

        # 1. Read txt file containing .npy paths
        self.detection_dic = {}       # vidname -> list of clip paths
        self.class_to_videos = {}     # class_name -> list of unique vidnames
        h5f_path = os.path.join(opt.hdf5_path, opt.mode + '.txt')
        
        with open(h5f_path) as f:
            detections = [line.strip() for line in f.readlines() if line.strip()]
            
        # Gather clips for each video and group videos by class
        for npy_detect in detections:
            vidname = get_vid_name(npy_detect)
            clsname = get_class_name(npy_detect)

            if vidname in self.detection_dic:
                self.detection_dic[vidname].append(npy_detect)
            else:
                self.detection_dic[vidname] = [npy_detect]

            if clsname not in self.class_to_videos:
                self.class_to_videos[clsname] = []
            if vidname not in self.class_to_videos[clsname]:
                self.class_to_videos[clsname].append(vidname)

        self.classes = sorted(list(self.class_to_videos.keys()))
        
        # --- TINH TOAN TRONG SO CAN BANG MEM (SMOOTH SAMPLING WEIGHTS: sqrt(N_c)) ---
        raw_weights = [math.sqrt(len(self.class_to_videos[cls])) for cls in self.classes]
        total_weight = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
        self.class_weights = [w / total_weight for w in raw_weights]

        print(f"[{opt.mode.upper()}] Da load {len(detections)} clips tu {len(self.detection_dic)} videos qua {len(self.classes)} lop nhac cu:")
        for cls, w in zip(self.classes, self.class_weights):
            num_vids = len(self.class_to_videos[cls])
            print(f"  - Lop '{cls}': {num_vids:2d} videos | Xac suat can bang mem (sqrt): {w*100:.2f}%")

        # 2. Initialize Transforms
        if opt.mode == 'val':
            vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        elif opt.preserve_ratio:
            vision_transform_list = [transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()]
        else:
            vision_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        
        if opt.subtract_mean:
            vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.vision_transform = transforms.Compose(vision_transform_list)

        # 3. Load Additional Scene Images (if specified)
        if opt.with_additional_scene_image:
            scene_h5f_path = os.path.join(opt.scene_path)
            h5f = h5py.File(scene_h5f_path, 'r')
            self.scene_images = h5f['image'][:]

    def __getitem__(self, index):
        # --- CAN BANG MEM (SMOOTH / TEMPERATURE SAMPLING) CHO TRAINING ---
        # Su dung xac suat ti le thuan voi sqrt(N_c) de:
        # 1. Tang tan suat xuat hien cua lop hiem (3-4 video) ma khong lam overfit.
        # 2. Van bao toan thoi luong hoc day du cho cac lop da so (21 video) de giu vung diem SDR.
        if self.opt.mode == 'train' and len(self.classes) >= self.NUM_PER_MIX:
            selected_classes = []
            available_classes = list(self.classes)
            available_weights = list(self.class_weights)

            for _ in range(self.NUM_PER_MIX):
                chosen_cls = random.choices(available_classes, weights=available_weights, k=1)[0]
                selected_classes.append(chosen_cls)
                # Loai lop da chon de dam bao mix giua 2 nhac cu khac nhau
                chosen_idx = available_classes.index(chosen_cls)
                available_classes.pop(chosen_idx)
                available_weights.pop(chosen_idx)

            videos2Mix = [random.choice(self.class_to_videos[c]) for c in selected_classes]
        else:
            # Fallback neu so lop nho hon NUM_PER_MIX hoac khi val/test
            all_vids = list(self.detection_dic.keys())
            if len(all_vids) >= self.NUM_PER_MIX:
                videos2Mix = random.sample(all_vids, self.NUM_PER_MIX)
            else:
                videos2Mix = random.choices(all_vids, k=self.NUM_PER_MIX)
        
        clip_det_paths = [None for _ in range(self.NUM_PER_MIX)]
        clip_det_bbs = [None for _ in range(self.NUM_PER_MIX)]
        
        for n in range(self.NUM_PER_MIX):
            # Randomly sample a clip from the video
            clip_det_paths[n] = random.choice(self.detection_dic[videos2Mix[n]]) 
            # Load the bbs for the clip and sample one from each class
            clip_det_bbs[n] = sample_object_detections(np.load(clip_det_paths[n]))

        audios = [None for _ in range(self.NUM_PER_MIX)]
        objects_visuals = []
        objects_labels = []
        objects_audio_mag = []
        objects_audio_phase = []
        objects_vids = []
        objects_audio_mix_mag = []
        objects_audio_mix_phase = []

        for n in range(self.NUM_PER_MIX):
            # Generate a unique video id
            vid = random.randint(1, 100000000000)
            
            # --- AUDIO PROCESSING ---
            audio_path = get_audio_path(clip_det_paths[n])
            audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)
            audio_segment = sample_audio(audio, self.audio_window)
            
            if self.opt.enable_data_augmentation and self.opt.mode == 'train':
                audio_segment = augment_audio(audio_segment)
                
            audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)            
            
            # Make a copy of the audio to mix later
            audios[n] = audio_segment
            detection_bbs = clip_det_bbs[n]

            # --- VISUAL PROCESSING (FRAMES) ---
            for i in range(detection_bbs.shape[0]):
                frame_id = detection_bbs[i,0]
                frame_path = get_frame_path(clip_det_paths[n], frame_id)
                
                # Make the label start from 0
                label = detection_bbs[i,1] - 1
                object_image = Image.open(frame_path).convert('RGB').crop((
                    detection_bbs[i,-4], detection_bbs[i,-3], 
                    detection_bbs[i,-2], detection_bbs[i,-1]
                ))
                
                if self.opt.enable_data_augmentation and self.opt.mode == 'train':
                    object_image = augment_image(object_image)
                    
                objects_visuals.append(self.vision_transform(object_image).unsqueeze(0))
                objects_labels.append(label)
                
                # Make a copy of the audio spec for each object
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)
            
            # --- SCENE IMAGE PROCESSING ---
            if self.opt.with_additional_scene_image:
                scene_image_path = random.choice(self.scene_images)
                scene_image = Image.open(scene_image_path).convert('RGB')
                
                if self.opt.enable_data_augmentation and self.opt.mode == 'train':
                    scene_image = augment_image(scene_image)
                    
                objects_visuals.append(self.vision_transform(scene_image).unsqueeze(0))
                
                # Use padded label for scene image
                objects_labels.append(self.opt.number_of_classes - 1)
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)

        # --- MIX AUDIO ---
        audio_mix = np.asarray(audios).sum(axis=0) / self.NUM_PER_MIX
        audio_mix_mag, audio_mix_phase = generate_spectrogram_magphase(audio_mix, self.stft_frame, self.stft_hop)
        
        # Make a copy of mixed audio spec for each object
        for n in range(self.NUM_PER_MIX):
            detection_bbs = clip_det_bbs[n]
            for i in range(detection_bbs.shape[0]):
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))
            
            if self.opt.with_additional_scene_image:
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))
        
        # --- EMPTY BBOX FIX ---
        if len(objects_visuals) == 0:
            print(f"Warning: Empty detection found! Resampling data...")
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)

        # --- PACK DATA ---
        data = {
            'visuals': np.vstack(objects_visuals),          # Detected objects
            'labels': np.vstack(objects_labels),            # Labels for each object
            'audio_mags': np.vstack(objects_audio_mag),     # Audio spectrogram magnitude
            'audio_mix_mags': np.vstack(objects_audio_mix_mag), 
            'vids': np.vstack(objects_vids)                 # Unique video indexes
        }

        # Include phase information only for validation/testing
        if self.opt.mode in ['val', 'test']:
            data['audio_phases'] = np.vstack(objects_audio_phase)
            data['audio_mix_phases'] = np.vstack(objects_audio_mix_phase)
            
        return data

    def __len__(self):
        if self.opt.mode == 'train':
            return self.opt.batchSize * self.opt.num_batch
        elif self.opt.mode == 'val':
            return self.opt.batchSize * self.opt.validation_batches

    def name(self):
        return 'AudioVisualMUSICDataset'
