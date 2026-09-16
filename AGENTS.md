# AGENTS.md

Ghi chú cho các phiên làm việc sau trên repo `audio_visual`.

## Bối cảnh

Repo nghiên cứu tách nguồn audio-visual (co-separation). Ban đầu là mô hình 5 nhánh của tác
giả gốc; hiện đã tối giản còn **2 nhánh**: `net_visual` (Resnet18) + `net_unet`
(AudioVisual5/7layerUNet). Loss duy nhất là co-separation mask loss.

`PROJECT_ANALYSIS.md` ở gốc repo là tài liệu tham chiếu đầy đủ (kiến trúc, luồng dữ liệu,
kích thước tensor, loss, optimizer, vòng lặp, validation, checkpoint, inference, metric, nợ
kỹ thuật). **Đọc file đó trước khi phân tích lại từ đầu.**

## Chạy nhanh để kiểm tra

Không có `test.py`. Muốn smoke test cần dữ liệu theo đúng layout:

```
<data_path>/
├── yolo_top_detections/<bất kỳ>/<video>_clip_<n>.npy   # [frame_id, class, 0, 0, x1, y1, x2, y2]
├── reshape_11025/<bất kỳ>/<video>_clip_<n>.wav
└── frame/<bất kỳ>/<video>_clip_<n>/<frame_id:06d>.png
```

Tên thư mục gốc phải đúng ba tên trên; thư mục con tuỳ ý (`glob(..., recursive=True)`).

```bash
python train.py --name smoke --model audioVisual --data_path <data_path> \
  --auto_split --batchSize 1 --nThreads 0 --num_batch 4 \
  --unet_num_layers 5 --optimizer sgd --gpu_ids -1 --checkpoints_dir ./checkpoints
```

Bắt buộc đặt `--num_batch` nhỏ: mặc định là 30000. `--gpu_ids -1` là cách duy nhất chọn CPU.

## Cạm bẫy đã gặp

- **Phải duyệt `dataset_loader`, không phải `dataset`.** Duyệt thẳng `Dataset` trả dict numpy
  thô; `audio_mix_mags.size(0)` sẽ ném `TypeError: 'int' object is not callable` vì `.size`
  của `ndarray` là số, không phải hàm.
- **`--tensorboard True/False` không hoạt động.** `argparse` với `type=bool` biến mọi chuỗi
  khác rỗng thành `True`. Muốn tắt thì bỏ hẳn cờ. Áp dụng cho cả `validation_on`,
  `subtract_mean`, `log_freq`, `measure_time`, `preserve_ratio`, `enable_data_augmentation`.
- **`opt.hdf5_path` thực chất là thư mục split**, không phải file HDF5. `train.py` gán
  `opt.hdf5_path = split_dir`. Tên tham số là di sản.
- **`opt_val = copy.copy(opt)` là bắt buộc**, không phải phong cách. Gán `opt.mode = 'val'` lên
  object dùng chung sẽ tắt augmentation của tập train bên trong worker.
- **Checkpoint phải chứa `best_err` cho mọi tag**, không chỉ `latest`. `save_latest_freq` chạy
  trước validation trong cùng batch, nên sidecar chỉ ghi lúc `latest` sẽ mang `best_err` cũ và
  lần resume sau ghi đè `unet_best.pth` bằng model tệ hơn.
- **`utils/utils.py` import `distutils.command.clean`** — chạy được chỉ nhờ shim của
  `setuptools`. Trên môi trường không có shim sẽ `ImportError` ngay khi import `utils`.
- **`models/networks.py` có `NameError` ẩn**: `AudioVisual7layerUNet_Audio` và
  `AudioVisual5layerUNet_Audio` tham chiếu `audio_conv7feature` không tồn tại. Không được
  `ModelBuilder` gọi nên lỗi đang ẩn.
- Val loader dùng `num_workers=2` cố định (không theo `--nThreads`). TensorBoard ghi ra `runs/`
  **trong thư mục hiện hành**, không phải trong `checkpoints_dir`.

## Thông tin đã đo, không cần đo lại

- STFT: `n_fft=1022, hop=256, window=65535` → phổ `(1, 512, 256)`. Sau `warpgrid` (log-freq)
  → `(1, 256, 256)`.
- `net_visual(4,3,224,224)` → `(4,512,1,1)`. UNet 5 và 7 tầng đều trả `(4,1,256,256)` ∈ (0,1).
- Dataset trả `visuals (O,3,224,224)`, `audio_mags (O,1,512,256)`, `vids (O,1)`; `audio_phases`
  chỉ có ở `mode in ('val','test')`.
- Mix audio là **trung bình cộng** `sum(audios)/NUM_PER_MIX`, không chuẩn hoá SNR — vì thế
  `gt_mask` mới cần `clamp_(0., 5.)`.
- Split chia **theo video** (`get_vid_name` cắt tại `_clip_`) để tránh rò rỉ dữ liệu. Đã kiểm
  chứng không có video nào nằm cả hai phía.