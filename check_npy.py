import os
import glob
import numpy as np

def check_empty_npy():
    target_dir = "/media/data4/home/lanlt/ducanh/dataset_av/yolo_top_detections/MUSIC_solo/top_detections_two_classes/"
    
    search_path = os.path.join(target_dir, "**", "*.npy")
    npy_files = glob.glob(search_path, recursive=True)
    
    if len(npy_files) == 0:
        print(f"Khong tim thay file .npy nao trong thu muc: {target_dir}")
        return

    empty_files = []
    corrupted_files = []
    
    print(f"Dang tien hanh quet {len(npy_files)} files...\n")
    
    for file_path in npy_files:
        try:
            data = np.load(file_path)
            
            if data.size == 0 or data.shape[0] == 0:
                empty_files.append(file_path)
                
        except Exception as e:

            corrupted_files.append((file_path, str(e)))

    print("=" * 50)
    print("DANH SACH CAC FILE TRONG (EMPTY DETECTIONS):")
    print("=" * 50)
    for f in empty_files:
        print(f)
        
    if len(corrupted_files) > 0:
        print("\n" + "=" * 50)
        print("DANH SACH CAC FILE BI HONG (CORRUPTED):")
        print("=" * 50)
        for f, err in corrupted_files:
            print(f"{f} (Loi: {err})")
            
    print("\n" + "=" * 50)
    print("TONG KET:")
    print(f"- Tong so file da quet : {len(npy_files)}")
    print(f"- So file binh thuong  : {len(npy_files) - len(empty_files) - len(corrupted_files)}")
    print(f"- So file bi TRONG     : {len(empty_files)}")
    print(f"- So file b? HONG      : {len(corrupted_files)}")
    print("=" * 50)

if __name__ == '__main__':
    check_empty_npy()