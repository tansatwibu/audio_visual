import os
import random
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Chia dataset thanh train/val/test txt files")
    parser.add_argument('--npy_dir', type=str, required=True, help="Thu muc goc chua cac file .npy (Vi du: /media/data4/home/lanlt/ducanh/dataset_av/yolo_top_detections)")
    parser.add_argument('--out_dir', type=str, default='.', help="Thu muc de luu cac file .txt ket qua")
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    args = parser.parse_args()

    # Tim toan bo file .npy trong thu muc va cac thu muc con
    search_path = os.path.join(args.npy_dir, '**', '*.npy')
    npy_files = glob.glob(search_path, recursive=True)

    if len(npy_files) == 0:
        print(f"Loi: Khong tim thay file .npy nao trong {args.npy_dir}")
        return

    print(f"Da tim thay {len(npy_files)} file .npy.")

    # Xao tron danh sach
    random.seed(42) # Co dinh seed de dam bao tinh tai lap
    random.shuffle(npy_files)

    # Tinh toan so luong tung tap
    total = len(npy_files)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_files = npy_files[:train_end]
    val_files = npy_files[train_end:val_end]
    test_files = npy_files[val_end:]

    # Ghi ra file
    os.makedirs(args.out_dir, exist_ok=True)

    def write_to_file(filepath, file_list):
        with open(filepath, 'w') as f:
            for item in file_list:
                f.write(f"{item}\n")

    train_path = os.path.join(args.out_dir, 'train_yolo_solo.txt')
    val_path = os.path.join(args.out_dir, 'val_yolo_solo.txt')
    test_path = os.path.join(args.out_dir, 'test_yolo_solo.txt')

    write_to_file(train_path, train_files)
    write_to_file(val_path, val_files)
    write_to_file(test_path, test_files)

    print("Hoan thanh chia dataset!")
    print(f"- {train_path}: {len(train_files)} files ({len(train_files)/total*100:.1f}%)")
    print(f"- {val_path}: {len(val_files)} files ({len(val_files)/total*100:.1f}%)")
    print(f"- {test_path}: {len(test_files)} files ({len(test_files)/total*100:.1f}%)")

if __name__ == '__main__':
    main()
