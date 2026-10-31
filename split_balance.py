import os
import glob
import random
from collections import defaultdict
import itertools

def main():
    npy_dir = "/media/data4/home/lanlt/ducanh/dataset_av/yolo_top_detections/MUSIC_solo/top_detections_two_classes"
    out_dir = "/media/data4/home/lanlt/ducanh/dataset_av/hdf5_split_balance"
    

    npy_files = glob.glob(os.path.join(npy_dir, '**', '*.npy'), recursive=True)
    
    class_dict = defaultdict(list)
    for f in npy_files:
        basename = os.path.basename(f)
        if "_video_" in basename:
            class_name = basename.split("_video_")[0].lower()
            class_dict[class_name].append(f)
            
    print(f"Da tim thay {len(class_dict.keys())} lop nhac cu:")
    
    train_list, val_list, test_list = [], [], []
    train_dict = defaultdict(list)
    

    random.seed(42)
    for class_name, files in class_dict.items():
        random.shuffle(files)
        total = len(files)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        train_dict[class_name] = train_files
        val_list.extend(val_files)
        test_list.extend(test_files)
        
        print(f"  - {class_name.ljust(15)}: Tong {total:3d} | Train: {len(train_files):3d}")


    max_train_samples = max(len(files) for files in train_dict.values())
    print(f"\n=> Se oversample tat ca cac lop Train len muc: {max_train_samples} clips!")

    for class_name, files in train_dict.items():
        if len(files) == 0: continue
        
        if len(files) < max_train_samples:
            oversampled = list(itertools.islice(itertools.cycle(files), max_train_samples))
            train_list.extend(oversampled)
        else:
            train_list.extend(files)


    random.shuffle(train_list)
    random.shuffle(val_list)
    
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        f.write("\n".join(train_list))
    with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
        f.write("\n".join(val_list))
    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        f.write("\n".join(test_list))

    print("\nHOAN THANH!")
    print(f"Tap Train can bang: {len(train_list)} files (Tat ca deu chiem ti trong bang nhau)")
    print(f"Tap Val: {len(val_list)} files")
    print(f"Tap Test goc: {len(test_list)} files")

if __name__ == '__main__':
    main()