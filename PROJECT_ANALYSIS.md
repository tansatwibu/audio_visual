# PROJECT_ANALYSIS — Audio-Visual Co-Separation (hai nhánh)

Phân tích kỹ thuật repo `audio_visual`: kiến trúc, luồng dữ liệu, kích thước tensor, loss,
optimizer, vòng lặp huấn luyện, validation, checkpoint, inference và cách tính metric.

> Mọi nhận định được rút ra trực tiếp từ source code trong repo.
> Những chỗ suy luận (do thiếu dữ liệu/tham số runtime) đều ghi rõ là **suy luận**.
>
> **Trạng thái repo lúc phân tích:** nhánh `twobranch-audio-visual-coseparation`,
> commit `dcf4a0c`. Code đã tối giản còn **đúng hai nhánh** (visual + UNet) và **đã chạy
> thông end-to-end** bằng dữ liệu tổng hợp (xem mục 12).

---

## 1. Tổng quan repo

| File | Vai trò |
|---|---|
| `train.py` (504 dòng) | Entry point duy nhất. Build model, optimizer, loss, vòng lặp train + validation + checkpoint. |
| `script.sh` | Lệnh chạy thật trên server (đường dẫn dữ liệu + siêu tham số). |
| `options/base_options.py` | Tham số chung (data path, STFT, batch size, gpu, checkpoint dir). |
| `options/train_options.py` | Tham số train (loss weight, LR, validation, unet, optimizer). |
| `options/test_options.py` | Tham số cho chế độ test. **Chưa có script nào dùng tới.** |
| `models/models.py` | `ModelBuilder` — factory tạo **2** subnet. |
| `models/networks.py` | `Resnet18`, `AudioVisual5layerUNet`, `AudioVisual7layerUNet`, `weights_init`, `unet_conv/upconv`. Còn nhiều class cũ không dùng. |
| `models/audioVisual_model.py` | `AudioVisualModel` — ghép 2 nhánh thành pipeline forward. |
| `models/criterion.py` | `L1Loss`, `L2Loss`, `BCELoss`, `CELoss`, `TripletLoss`, `TripletLossCosine`. |
| `data/ducanh_audioVisual_dataset.py` | Dataset **đang dùng** (không có pose, có `--data_path`). |
| `data/audioVisual_dataset.py` | Biến thể cũ: có pose feature, hard-code đường dẫn tuyệt đối. **Không dùng.** |
| `data/hai_audioVisual_dataset.py` | Biến thể cũ khác. **Không dùng.** |
| `data/custom_dataset_data_loader.py` | `DataLoader` với `collate_fn = object_collate`. **Không dùng** (train.py tự tạo loader). |
| `utils/utils.py` | `warpgrid`, `magnitude2heatmap`, `istft_reconstruction`, `object_collate`, hàm mix theo SNR. |
| `utils/viz.py` | `HTMLVisualizer` — xuất bảng HTML có ảnh + audio. |
| `dataset/` | `.h5` và `.txt` mẫu chứa **đường dẫn** `.npy` của môi trường tác giả. Chỉ để tham chiếu. |

Điểm quan trọng: repo **không có `test.py` / `inference.py` / `metrics.py`**. Toàn bộ phần
inference và metric chỉ mô tả ở mức "suy ra từ forward pass" — xem mục 10 và 11.

Chạy: `python train.py [options]`, hoặc `bash script.sh`. `BaseOptions.parse()` in toàn bộ
option ra stdout và ghi `opt.txt` vào `checkpoints_dir/name/`.

`models.py::ModelBuilder` chỉ còn đúng `build_visual` và `build_unet`. Phần di sản của mô
hình cũ nằm rải ở `models/criterion.py` (các lớp `CELoss`, `TripletLoss`, `TripletLossCosine`
không còn ai gọi) và trong `options/*.py` (các cờ classifier/vocal/facial — xem mục 13 #6).
`train.py` tự tạo `DataLoader` trong `create_loader` (`train.py:79`) nên
`data/custom_dataset_data_loader.py` và `data/data_loader.py` hoàn toàn không nằm trên đường
chạy.

---

## 2. Architecture

### 2.1 Sơ đồ tổng thể

```
   visuals ────────►┌───────────────────────────┐
   (B,3,224,224)    │ net_visual (Resnet18)     │──► visual_feature
                    │ maxpool, with_fc=False    │    (B,512,1,1)
                    └───────────────────────────┘            │
                                                             │ repeat
   audio_mags ─────►┌───────────────────────────┐            │
   (B,1,256,256)    │ net_unet                  │◄───────────┘
   (log, detached)  │ UNet 5/7 tầng, ngf=64     │
                    │ skip + visual bottleneck  │──► pred_mask
                    └───────────────────────────┘    (B,1,256,256), Sigmoid

   gt_mask = audio_mags / audio_mix_mags   (clamp 0..5)   ← nhãn, không qua mạng

   pred_spectrogram = audio_mix_mags * pred_mask           ← chỉ dùng để trực quan
```

`AudioVisualModel.__init__(self, nets, opt)` nhận tuple **2 net** theo đúng thứ tự
`(net_visual, net_unet)` và lưu vào `self.net_visual`, `self.net_unet`.

### 2.2 Các subnet

| Net | Class | Input | Output | Ghi chú |
|---|---|---|---|---|
| `net_visual` | `Resnet18(pool_type='maxpool', with_fc=False)` | `(B,3,224,224)` | `(B,512,1,1)` | `conv1` được thay để nhận số channel tuỳ biến; trọng số pretrained ResNet18. `fc_out=256` truyền vào nhưng **vô tác dụng** vì `with_fc=False`. |
| `net_unet` | `AudioVisual7layerUNet` hoặc `AudioVisual5layerUNet` | `x=(B,1,256,256)`, `visual_feat=(B,512,1,1)` | `(B,1,256,256)`, Sigmoid | `unet_ngf=64`. Chọn qua `--unet_num_layers` (5 hoặc 7). |

Classifier / vocal / facial / pose **đã bị loại bỏ hoàn toàn**: không được build, không nằm
trong optimizer, không nằm trong checkpoint, không xuất hiện trong `forward`.

`AudioVisualModel.forward` (`models/audioVisual_model.py:26`):

```python
audio_mix_mags = audio_mix_mags + 1e-10          # tránh chia 0 khi tính gt_mask
if self.opt.log_freq:                            # warp trục tần số sang log-scale
    audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
    audio_mags     = F.grid_sample(audio_mags, grid_warp)
gt_masks = audio_mags / audio_mix_mags
gt_masks.clamp_(0., 5.)
visual_feature  = self.net_visual(Variable(visuals, requires_grad=False))
audio_log_mags  = torch.log(audio_mix_mags).detach()
mask_prediction = self.net_unet(audio_log_mags, visual_feature)
```

Hai chi tiết cần nhớ:

- **`visuals` được đặt `requires_grad=False`** khi vào `net_visual`. Đây là input không cần
  grad chứ không phải chặn nhánh visual: gradient vẫn chảy qua `net_visual` bình thường. Muốn
  thực sự tắt grad cho tham số phải dùng `--freeze_visual` (`train.py:374`).
- **`audio_log_mags` bị `.detach()`** trước khi vào UNet. Nghĩa là gradient chỉ chảy vào nhánh
  mask, không đẩy mạng đi "mã hoá lại" chính hỗn hợp âm thanh. UNet *nhận* phổ hỗn hợp làm
  input nhưng *không* được tối ưu để tái tạo nó.

### 2.3 UNet 5 tầng — chi tiết (ngf=64, input 256×256)

`unet_conv` = `Conv2d(k=4, s=2, p=1) → BatchNorm2d → LeakyReLU(0.2)`.
`unet_upconv` = `ConvTranspose2d(k=4, s=2, p=1) → BatchNorm2d → ReLU`; tầng ngoài cùng thay
BN+ReLU bằng `Sigmoid`.

Số đo thực tế (chạy `models/networks.py` với tensor giả):

| Tầng | Out channels | H×W |
|---|---|---|
| conv1 | 64 | 128×128 |
| conv2 | 128 | 64×64 |
| conv3 | 256 | 32×32 |
| conv4 | 512 | 16×16 |
| conv5 | 512 | 8×8 |

Bottleneck: `visual_feat` được `repeat` lên `(B,512,8,8)` rồi `cat` với `conv5` theo channel
→ `(B,1024,8,8)`.

Decoder: `upconv1` (1024→512) → `upconv2` (1024→256) → `upconv3` (512→128) → `upconv4`
(256→64) → `upconv5` (128→1, Sigmoid). Các skip connection lấy `conv4..conv1` tương ứng.
Kết quả: `(B,1,256,256)`, giá trị trong khoảng `(0,1)` — đúng như đã đo.

### 2.4 UNet 7 tầng

Giống hệt 5 tầng nhưng encoder sâu thêm 2 tầng `conv6`, `conv7` (512 channel, 4×4 rồi 2×2)
và decoder có 7 tầng `upconv1..upconv7`. `upconv1` nhận `ngf*8 + visual_dim = 512 + 512 = 1024`
channel. Kết quả vẫn `(B,1,256,256)`.

`--unet_num_layers` chỉ nhận `5` hoặc `7`; giá trị khác bị `build_unet` từ chối bằng
`ValueError`.

### 2.5 ⚠️ Nợ kỹ thuật trong `models/networks.py`

File 803 dòng nhưng phần lớn là **code chết**:

- Ba khối `unet_conv` / `unet_upconv` / `create_conv` / `class Resnet18` bị comment ở đầu file
  (dòng 7–108), trùng lặp với bản đang dùng ở dòng 378+.
- `AudioVisual7layerUNet_Audio` (dòng 481) và `AudioVisual5layerUNet_Audio` (dòng 535) có
  `forward` tham chiếu `audio_conv7feature` **không tồn tại trong scope** → sẽ `NameError`
  nếu gọi. Chúng không được `ModelBuilder` dùng nên lỗi này đang ẩn.
- Còn nhiều biến thể có pose (`AudioPose7layerUNet`, `AudioVisualPose7layerUNet`) và các biến
  thể UNet khác không còn ai gọi.
- `Resnet18.forward_multiframe` có một `return x` không thể tới.
- Import thừa: `functools`, `collections.OrderedDict`.

Không ảnh hưởng lúc chạy, nhưng nên dọn ở lần refactor sau.

---

## 3. Data flow

### 3.1 Chuẩn bị split: `--auto_split`

`build_split_lists` (`train.py:38`) quét `data_path/yolo_top_detections/**/*.npy`, gom theo
`get_vid_name()`, rồi chia train/val **theo video**. Việc gom theo video là bắt buộc: hai
clip của cùng một buổi biểu diễn chia về hai phía sẽ làm rò rỉ dữ liệu (cùng nhạc sĩ, cùng
nhạc cụ, cùng phòng thu).

`get_vid_name` cắt tên clip tại `_clip_`:

```
dan_bau_video_10_clip_001  ->  dan_bau_video_10
```

Hàm này được import trực tiếp từ dataset (`from data.ducanh_audioVisual_dataset import
get_vid_name`) để split và loader luôn nhất quán về định nghĩa "một video".

Split ghi ra `split_dir` mặc định `checkpoints_dir/name/splits/`, gồm `train.txt` và `val.txt`,
mỗi dòng một đường dẫn `.npy` tuyệt đối.

### 3.2 Từ split file → danh sách clip

`AudioVisualMUSICDataset.initialize` (`data/ducanh_audioVisual_dataset.py:113`):

1. Đọc `os.path.join(opt.hdf5_path, opt.mode + '.txt')`. `train.py:324` gán
   `opt.hdf5_path = split_dir`, nên `hdf5_path` ở đây thực chất là thư mục split — tên tham
   số là di sản của bản gốc dùng file HDF5.
2. Gom clip theo video id vào `self.detection_dic`.
3. `transforms`: `val` → `Resize((224,224))`; `train` → `Resize((256,256))` + `RandomCrop(224)`
   (hoặc `Resize(256)+RandomCrop(224)` nếu `preserve_ratio`), cộng `Normalize(ImageNet mean/std)`
   nếu `subtract_mean`.

Đường dẫn frame/audio **được suy ra từ đường dẫn `.npy`**, đây là điểm sửa quan trọng so với
hai biến thể dataset cũ:

```python
get_audio_path(npy): dirname.replace("yolo_top_detections", "reshape_11025") → <clip>.wav
get_frame_path(npy, frame_id): dirname.replace("yolo_top_detections", "frame")
                               → <clip>/<frame_id:06d>.png
```

Nhờ vậy **chỉ cần một `--data_path`** trỏ tới thư mục chứa đồng thời `frame/`,
`reshape_11025/` và `yolo_top_detections/`. Không còn đường dẫn tuyệt đối hard-code.

### 3.3 Một mẫu `__getitem__`

```
random.sample(video_ids, NUM_PER_MIX=2)          # 2 video khác nhau
for n in 0..1:
    clip_det_paths[n] = random.choice(detection_dic[video[n]])
    clip_det_bbs[n]   = sample_object_detections(np.load(clip_det_paths[n]))
```

`sample_object_detections` nhóm detection theo class id rồi lấy **ngẫu nhiên 1 detection mỗi
class**, nên số object mỗi clip thay đổi theo dữ liệu (với dataset 2 class → ~2 object/clip;
tổng `O ≈ 4` object cho 2 clip).

Với mỗi object:

- `audio` nạp bằng `librosa.load(sr=audio_sampling_rate)`, cắt cửa sổ `audio_window` mẫu
  (`sample_audio` tự `np.tile` nếu audio ngắn hơn cửa sổ).
- `augment_audio` (chỉ khi `enable_data_augmentation` và `mode == 'train'`): scale biên độ
  ngẫu nhiên 0.5–1.5 rồi clip về `[-1, 1]`.
- `augment_image`: đổi ngẫu nhiên Brightness/Color.
- Ảnh crop theo bounding box → `vision_transform` → `(3,224,224)`.

Sau đó trộn audio:

```python
audio_mix = np.asarray(audios).sum(axis=0) / self.NUM_PER_MIX
```

tức **trung bình cộng**, không chuẩn hoá theo SNR. Hệ quả: năng lượng hỗn hợp giảm khi tăng
`--num_per_mix`, và `gt_mask = audio_mags / audio_mix_mags` không còn nằm trong `[0,1]` một
cách tự nhiên — đây chính là lý do có `clamp_(0., 5.)` ở mục 2.2.

`data` trả về là **dict numpy**, không phải tensor:

| Key | Shape | Dtype |
|---|---|---|
| `visuals` | `(O,3,224,224)` | float32 |
| `labels` | `(O,1)` | — |
| `audio_mags` | `(O,1,512,256)` | float32 |
| `audio_mix_mags` | `(O,1,512,256)` | float32 |
| `vids` | `(O,1)` | int |
| `audio_phases`, `audio_mix_phases` | `(O,1,512,256)` | **chỉ có ở `mode in ('val','test')`** |

`labels` vẫn được sinh ra nhưng **không còn ai đọc** (classifier đã bị loại).

### 3.4 `__len__` và số batch

```python
def __len__(self):
    if self.opt.mode == 'train': return self.opt.batchSize * self.opt.num_batch
    elif self.opt.mode == 'val': return self.opt.batchSize * self.opt.validation_batches
```

Vì `__getitem__` lấy mẫu ngẫu nhiên nên "epoch" ở đây chỉ là một vòng lặp có độ dài cố định;
`--niter` mặc định 1. Số batch thực chạy = `num_batch * niter`.

**Hệ quả thực tế:** `--num_batch` mặc định là 30000. Khi test nhanh mà quên đặt `--num_batch`,
chương trình sẽ chạy 30000 batch chứ không dừng sớm.

### 3.5 Collate

`object_collate` (`utils/utils.py`) xử lý dict lồng nhau: mỗi field numpy được
`torch.cat([torch.from_numpy(b) for b in batch], 0)` — **nối theo dim 0 chứ không stack**.
Với `batchSize=1` thì không khác biệt; với `batchSize>1` các object của mọi mẫu bị trộn thành
một danh sách phẳng dài `B*O`. Loss theo clip vẫn hoạt động vì nó gom lại theo `vids`.

---

## 4. Tensor dimensions

### 4.1 Tham số âm thanh (mặc định)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `audio_sampling_rate` | 11025 Hz | tần số lấy mẫu |
| `audio_window` | 65535 | số mẫu mỗi đoạn (~5.94 s) |
| `stft_frame` (n_fft) | 1022 | cửa sổ STFT |
| `stft_hop` | 256 | bước nhảy STFT |

Số khung thời gian: `65535 / 256 + 1 ≈ 256`. Số bin tần số: `1022/2 + 1 = 512`.
Đã đo thực tế: `generate_spectrogram_magphase` trả `(1, 512, 256)`.

`--log_freq` mặc định `True` → `warpgrid(B, 256, 256, warp=True)` nén 512 bin xuống
**256 bin** theo thang log (giống mel). Sau warp, mọi tensor phổ đều là `(B,1,256,256)`.

### 4.2 Bảng tensor xuyên suốt (`B` = batch, `O` = số object)

| Bước | Tensor | Shape |
|---|---|---|
| dataset trả | `visuals`, `audio_mags`, `audio_mix_mags`, `vids` | `(O,3,224,224)`, `(O,1,512,256)`, `(O,1,512,256)`, `(O,1)` |
| sau `object_collate` | như trên, thành Tensor | giữ nguyên shape (dim 0 = `B*O`) |
| sau warp | `audio_mags`, `audio_mix_mags` | `(O,1,256,256)` |
| `net_visual(visuals)` | `visual_feature` | `(O,512,1,1)` |
| `net_unet(log(mix), feat)` | `pred_mask` | `(O,1,256,256)`, ∈ (0,1) |
| nhãn | `gt_mask = mags / mix_mags` | `(O,1,256,256)`, ∈ [0,5] |
| trọng số | `weight = clamp(log1p(mix), 1e-3, 10)` | `(O,1,256,256)` hoặc `None` |
| gộp theo clip | `predicted_mask_list[v]` | `(1,256,256)` — **tổng** các object cùng `vid` |
| loss | `coseparation_loss` | scalar |

### 4.3 Vì sao `visual_feature` là `(B,512,1,1)`

`Resnet18.forward` với `pool_type='maxpool'` gọi `F.adaptive_max_pool2d(x, 1)` rồi
`x.view(x.size(0), -1, 1, 1)`. UNet sau đó `repeat(1, 1, H, W)` để broadcast lên đúng kích
thước bottleneck. Đã đo: `(4,512,1,1)`.

Nếu đổi `--visual_pool` sang `conv1x1`, `with_fc=True` và `fc_out=256` được dùng, nhưng
`create_conv(512, 128, ...)` hard-code 128 → cần kiểm tra lại tính nhất quán. Mặc định
`maxpool` không gặp vấn đề này.

### 4.4 Lưu ý về `--subtract_mean`

Mặc định `True` → ảnh được Normalize theo ImageNet. Vì ResNet18 dùng trọng số pretrained
ImageNet nên đây là lựa chọn đúng. Nếu tắt, phân phối input lệch khỏi pretrained và nhánh
visual sẽ học kém hơn ở giai đoạn đầu.

---

## 5. Loss

### 5.1 Co-separation loss — mục tiêu duy nhất

`get_coseparation_loss` (`train.py:264`):

1. `vids = output['vids'].squeeze(1).cpu().numpy()`; dựng `vid_index_dic` ánh xạ mỗi `vid`
   sang một chỉ số `0..V-1`. `V` = số clip trong batch (`= NUM_PER_MIX` khi `batchSize=1`).
2. Với mỗi object `i`: nếu clip chưa có mặt, khởi tạo `gt_mask_list[c] = gt_masks[i]`,
   `predicted_mask_list[c] = pred_mask[i]`, `weight_list[c] = weight[i]`; nếu đã có mặt thì
   **cộng dồn mask dự đoán**: `predicted_mask_list[c] += mask_prediction[i]`.
3. Nếu `mask_loss_type == 'BCE'`, clamp `predicted_mask_list` về `[0,1]`.
4. `loss_coseparation(predicted_mask_list, gt_mask_list, weight_list)`.

Điểm cốt lõi: **mask của tất cả object thuộc cùng một clip được cộng lại** rồi mới so với
`gt_mask` (mask thật của nguồn đó trong hỗn hợp). Đây là "co-separation": mạng phải chia
nguồn âm thanh cho từng người/biểu diễn, chứ không học nhị phân "có/không có âm thanh".
`gt_mask` chỉ lấy từ object đầu tiên của clip nên giả định **mọi object trong một clip cùng
nghe chung một nguồn audio** (đúng với bố cục dataset).

### 5.2 `BaseLoss` và xử lý danh sách

`models/criterion.py`:

```python
def forward(self, preds, targets, weight=None):
    if isinstance(preds, list):
        N = len(preds)
        errs = [self._forward(preds[n], targets[n],
                              preds[n].new_ones(1) if weight is None else weight[n])
                for n in range(N)]
        err = torch.mean(torch.stack(errs))
    elif isinstance(preds, torch.Tensor):
        if weight is None:
            weight = preds.new_ones(1)
        err = self._forward(preds, targets, weight)
    return err
```

Nhánh list **bắt buộc mỗi phần tử có weight riêng**: một tensor cỡ 1 dùng chung cho mọi `n`
sẽ lỗi `IndexError` ngay khi `N > 1`, còn `new_ones(1)` per-entry thì broadcast đúng lên
toàn mẫu bên trong `_forward`. Đây là bug đã được sửa ở commit `e93332e`.

`L1Loss._forward` = `mean(weight * |pred - target|)`, `L2Loss` tương tự với bình phương.
`BCELoss._forward` gọi thẳng `F.binary_cross_entropy(pred, target, weight=weight)`.

### 5.3 Trọng số theo bin tần số

Khi `--weighted_loss`):

```python
weight = torch.log1p(audio_mix_mags)      # log(1 + mag)
weight = torch.clamp(weight, 1e-3, 10)
```

Bin to → trọng số lớn hơn → mạng tập trung vào vùng có năng lượng thật. Clamp `1e-3` tránh
trọng số 0 ở vùng im lặng (nếu để 0 thì gradient vùng đó triệt tiêu hoàn toàn), clamp `10`
chặn một bin quá to chi phối loss. Khi tắt `--weighted_loss`, `weight = None` và loss dùng
trọng số đều 1.

### 5.4 Tổng loss trong train

`train.py:440-450`:

```python
coseparation_loss = get_coseparation_loss(...) * opt.coseparation_loss_weight
optimizer.zero_grad(); coseparation_loss.backward(); optimizer.step()
```

`--coseparation_loss_weight` mặc định **20**. Vì đây là loss duy nhất, hệ số này chỉ là phép
đổi thang gradient — nó tương đương nhân learning rate lên 20 lần, không phải cân bằng giữa
nhiều loss như ở bản gốc. Với `script.sh` thì đặt bằng `1`.

Loss classification và triplet/crossmodal **đã bị xoá khỏi code**, không còn đường bật lại.

---

## 6. Optimizer

`create_optimizer` (`train.py:90`):

```python
param_groups = [{'params': net_unet.parameters(), 'lr': opt.lr_unet}]
if not opt.freeze_visual:
    param_groups.insert(0, {'params': net_visual.parameters(), 'lr': opt.lr_visual})
```

| | Giá trị mặc định | `script.sh` |
|---|---|---|
| `lr_visual` | 0.0001 | 0.0001 |
| `lr_unet` | 0.001 | 0.001 |

UNet học nhanh gấp 10 lần visual, hợp lý vì visual là pretrained còn UNet train from scratch.

- `sgd`: `SGD(lr per group, momentum=beta1=0.9, weight_decay=1e-4)`.
- `adam`: `Adam(betas=(0.9, 0.999), weight_decay=1e-4)`.
- `--freeze_visual`: gọi `utils.set_requires_grad([net_visual], False)` **và** bỏ nhóm tham số
  visual khỏi optimizer. Kiểm chứng thực tế: khi bật, log chỉ in ra một giá trị LR thay vì hai.

LR decay: `if total_batches in opt.lr_steps: decrease_learning_rate(optimizer, decay_factor)`,
nhân toàn bộ LR với `decay_factor=0.1`. `script.sh` đặt `--lr_steps 60000 90000`, khớp với
`--num_batch 100000`.

Vì `total_batches` được lưu vào `training_state.pth` và khôi phục khi `--continue_train`,
mốc decay không bị lặp lại sau khi resume — đã kiểm chứng: chạy tiếp sau batch 4 với
`--lr_steps 2` giữ nguyên LR đã giảm.

---

## 7. Training loop

`train.py:424`:

```python
for epoch in range(1 + opt.epoch_count, opt.niter + 1):
    for i, data in enumerate(dataset_loader):
        model.zero_grad()
        output = model(data)
        coseparation_loss = get_coseparation_loss(...) * opt.coseparation_loss_weight
        optimizer.zero_grad()
        coseparation_loss.backward()
        optimizer.step()
        total_batches += 1
```

Thứ tự các hook trong một batch, theo `total_batches`:

| Điều kiện | Hành động |
|---|---|
| `% display_freq == 0` | in loss trung bình, ghi `data/coseparation_loss` lên TensorBoard; nếu `--measure_time` in 3 số thời gian |
| `% save_latest_freq == 0` | lưu `visual_latest.pth`, `unet_latest.pth`, `training_state.pth`; in LR |
| `% validation_freq == 0 and validation_on` | `model.eval()` → `display_val` → `model.train()`; nếu tốt hơn thì lưu `*_best.pth` |
| `total_batches in lr_steps` | giảm LR ×0.1 |

Một số ghi chú về cách viết hiện tại:

- **`model.zero_grad()` rồi `optimizer.zero_grad()`** — gọi hai lần, thừa nhưng vô hại.
- **Duyệt `dataset_loader` chứ không phải `dataset`.** Đây là bug đã sửa: duyệt thẳng
  `Dataset` sẽ trả dict **numpy** thô và `audio_mix_mags.size(0)` ném
  `TypeError: 'int' object is not callable` (vì `.size` của `ndarray` là một số, không phải
  hàm). Loader mới chạy `object_collate` để đổi sang Tensor.
- **`--measure_time`**: đo bằng `time.time()` kèm `cuda_synchronize()`. `cuda_synchronize()`
  tự kiểm tra `torch.cuda.is_available()` trước, nên chạy CPU với `--gpu_ids -1` không lỗi.
  Ba chỉ số: data loading, forward, backward.
- **`--tensorboard`**: `SummaryWriter(comment=opt.name)`, ghi ra `runs/` trong thư mục hiện
  hành. Hai scalar: `data/coseparation_loss` và `data/val_coseparation_loss`.
- **Không có `DataParallel`**: `model` được gọi trực tiếp. `gpu_ids[0]` chỉ dùng để chọn
  device, các GPU khác trong danh sách bị bỏ qua.

---

## 8. Validation

`display_val` (`train.py:225`):

```python
with torch.no_grad():
    for i, val_data in enumerate(dataset_val_loader):
        if i >= opt.validation_batches:
            if opt.validation_visualization:
                save_visualization(vis_rows, model(val_data), val_data, save_dir, opt)
            break
        output = model(val_data)
        coseparation_loss = get_coseparation_loss(...) * opt.coseparation_loss_weight
        coseparation_losses.append(coseparation_loss.item())
avg = sum(coseparation_losses) / len(coseparation_losses)
```

- Val loader được dựng từ một `copy.copy(opt)` với `mode='val'` (`train.py:348`). Bản copy
  nông là cần thiết: nếu gán `opt.mode='val'` lên chính object dùng chung, augmentation của
  tập train sẽ bị tắt luôn bên trong worker.
- Dataset val **luôn trả thêm `audio_phases` và `audio_mix_phases`** vì cần để tái tạo
  waveform khi trực quan hoá.
- Khi bật `--validation_visualization`, `validation_batches` của opt val được cộng thêm 1
  (`train.py:352`) để có đúng một batch dư nuôi `save_visualization`.
- Val loader dùng `num_workers=2` cố định, không theo `--nThreads`.
- **Chỉ số duy nhất là co-separation loss trung bình.** `accuracy`, `classifier loss`,
  crossmodal... đã bị loại cùng với classifier. Hàm trả về scalar này và cũng chính nó dùng
  để chọn best model.

`save_visualization` (`train.py:135`) ghi ra `checkpoints_dir/name/visualization/example-<j>/`:

| File | Nội dung |
|---|---|
| `mix.wav`, `mix.jpg` | hỗn hợp (ISTFT từ mag+phase gốc; ảnh từ mag đã warp) |
| `gt.wav`, `gtamp.jpg` | nguồn thật: `mag_mix * gt_mask_linear` rồi ISTFT |
| `pred.wav`, `predamp.jpg` | nguồn dự đoán: `mag_mix * pred_mask_linear` rồi ISTFT |
| `gtmask.jpg`, `predmask.jpg` | mask dạng ảnh xám |
| `weight.jpg` | chỉ có khi `--weighted_loss` |
| `index.html` | bảng HTML do `viz.HTMLVisualizer` sinh, kèm thẻ `<audio>` |

Chú ý: `save_dir` trong `display_val` là `os.path.join('.', opt.checkpoints_dir, opt.name,
'visualization')` — có tiền tố `'.'` thừa, nhưng vì `checkpoints_dir` trong `script.sh` là
đường dẫn tuyệt đối nên `./` bị bỏ qua và kết quả vẫn đúng.

Mask được unwarp về thang tuyến tính (`grid_sample` với `warp=False`) trước khi nhân với phổ
gốc, nên waveform tái tạo mới đúng trục tần số. Hai cảnh báo `UserWarning: Default
grid_sample and affine_grid behavior has changed to align_corners=False` là vô hại nhưng có
thể làm lệch nhẹ phép unwarp so với ý định ban đầu của bản gốc.

---

## 9. Checkpoint

### 9.1 Ghi

`save_checkpoint` (`train.py:108`) lưu vào `checkpoints_dir/name/`:

| File | Nội dung |
|---|---|
| `visual_latest.pth`, `unet_latest.pth` | `state_dict` của từng nhánh |
| `visual_best.pth`, `unet_best.pth` | `state_dict` tại lần val tốt nhất |
| `training_state.pth` | `net_visual`, `net_unet`, `optimizer`, `total_batches`, `best_err` |

`training_state.pth` được ghi cho **mọi** tag, không chỉ `latest`. Đây là bug đã sửa ở commit
`dcf4a0c`: trong cùng một batch, `save_latest_freq` chạy **trước** validation, nên nếu chỉ ghi
sidecar lúc `latest` thì nó sẽ mang `best_err` **trước** khi val cập nhật; lần resume sau đó
sẽ thấy `best_err` cũ (lớn hơn) và ghi đè `unet_best.pth` bằng một model tệ hơn. Đã kiểm
chứng: `training_state.pth` giờ giữ đúng `best_err` sau validation (2.84817 khớp với log).

`--save_latest_freq` mặc định 200, `script.sh` đặt 1000. Mỗi lần ghi tốn ~250 MB
(`training_state.pth` chứa cả hai state_dict + optimizer), nên tần suất này cần cân nhắc khi
đĩa chật.

### 9.2 Đọc (resume)

`train.py:409`:

```python
if opt.continue_train:
    state = torch.load(state_path, map_location=opt.device)
    net_visual.load_state_dict(state['net_visual']); net_unet.load_state_dict(state['net_unet'])
    optimizer.load_state_dict(state['optimizer'])
    total_batches = state['total_batches']; best_err = state['best_err']
```

- Resume đặt **trước vòng lặp**, nên `total_batches` và LR đã decay có hiệu lực cho toàn bộ
  run — không bị lặp mốc `lr_steps`. Đã kiểm chứng.
- File không tồn tại → `FileNotFoundError` với thông báo rõ ràng thay vì lỗi khó hiểu.
- `--epoch_count` mặc định 0; chỉ cần đổi nếu muốn nhảy số epoch hiển thị.
- Model được load **trước** `--continue_train` mới được load state, nên resume ghi đè cả
  `--weights_visual` / `--weights_unet` nếu cả hai cùng được truyền.

---

## 10. Inference

### 10.1 Hiện trạng

**Repo không có script inference.** `options/test_options.py` tồn tại nhưng không file nào
import nó. `dataset/` chỉ chứa dữ liệu mẫu. Vì vậy mục này mô tả đường suy luận hợp lý từ
chính forward pass và cảnh báo những chỗ cần cẩn thận khi viết.

### 10.2 Từ checkpoint → waveform

```
1. Nạp model:  net_visual = ModelBuilder().build_visual(...)
               net_unet   = ModelBuilder().build_unet(unet_num_layers=...)
               model = AudioVisualModel((net_visual, net_unet), opt)
               model.load_state_dict? → hiện KHÔNG có state_dict phẳng, phải load riêng
               từng nhánh từ visual_best.pth / unet_best.pth
2. Chuẩn bị 1 mẫu với mode='val' (cần audio_phases + audio_mix_phases để tái tạo)
3. model.eval(); with torch.no_grad(): out = model(data)
4. pred_mask_linear = grid_sample(out['pred_mask'], grid_unwrap)   # nếu --log_freq
5. pred_mag = out['audio_mix_mags'][j,0] * pred_mask_linear[j,0]
6. pred_wav = utils.istft_reconstruction(pred_mag, phase_mix[j,0], hop_length=stft_hop)
7. scipy.io.wavfile.write(path, opt.audio_sampling_rate, pred_wav)
```

`pred_spectrogram` trong output (`audio_mix_mags * pred_mask`) **không** dùng trực tiếp để
tái tạo, vì nó còn nằm trên thang log-frequency. Phải unwarp về thang tuyến tính trước — đúng
như `save_visualization` đang làm (`train.py:148-154`).

Một điểm dễ nhầm khi đọc code: `save_visualization` lấy **hai** nguồn phổ với hai vai trò khác
nhau (`train.py:137,141`).

- `mag_mix = batch_data['audio_mix_mags']` — phổ gốc **chưa warp**, còn 512 bin. Dùng để tái
  tạo waveform, vì `istft_reconstruction` cần đúng trục tần số (n_fft = 1022 → 512 bin).
- `mag_mix_ = outputs['audio_mix_mags']` — phổ **đã warp** về 256 bin. Dùng để nhân với mask và
  vẽ ảnh. Nhân mask (256 bin) với `mag_mix` (512 bin) sẽ lỗi shape, nên đây không phải trùng
  lặp thừa.

### 10.3 Vấn đề pha — giới hạn cố hữu

Model **chỉ dự đoán magnitude**. Khi tái tạo, code dùng **pha của hỗn hợp**
(`phase_mix`) cho mọi nguồn. Hệ quả:

- Việc tách nguồn bị chặn trên bởi chất lượng ước lượng pha. Với hai nguồn chồng lấn về thời
  gian–tần số, dùng chung một pha cho cả hai nguồn gây méo và rò rỉ nguồn kia.
- Đây là lựa chọn thiết kế của bản gốc (mask-based separation), không phải bug. Muốn tốt hơn
  phải dự đoán pha hoặc dùng complex spectrogram (`--spectrogram_type complex` trong
  `test_options.py` là di sản của hướng đó, chưa được cài đặt).

### 10.4 Vấn đề biên khi cắt cửa sổ

`audio_window = 65535` nhưng `sample_audio` chọn cửa sổ ngẫu nhiên từ toàn bộ file. Ở chế độ
inference dài (file nhiều phút), không thể tái tạo một lần; cần cắt thành nhiều cửa sổ trượt
rồi ghép (overlap-add). `test_options.py` có `--hop_size 0.05` ám chỉ hướng này nhưng **chưa
có code**. Trên thực tế, `save_visualization` lấy **một cửa sổ ngẫu nhiên duy nhất** và chỉ
minh hoạ, không tính metric trên toàn file.

---

## 11. Metric calculation

### 11.1 Metric đang có

**Chỉ một**: co-separation mask loss trung bình trên tập val, in ra dưới dạng

```
val coseparation loss: 0.1390
```

và ghi lên TensorBoard ở tag `data/val_coseparation_loss`.

Hiện tại `script.sh` dùng `--mask_loss_type L1`, nên đây là **L1 có trọng số theo bin**
(nếu `--weighted_loss`) giữa mask dự đoán đã cộng dồn theo clip và mask thật.

### 11.2 Metric dùng cho model selection

`best_err` khởi tạo `float("inf")`; sau mỗi lần validation, nếu `val_err < best_err` thì lưu
`*_best.pth` và cập nhật `best_err`. `best_err` nằm trong `training_state.pth` nên lựa chọn
best model nhất quán xuyên các lần resume.

### 11.3 Metric KHÔNG có

| Metric | Trạng thái |
|---|---|
| SDR / SI-SDR / SI-SNR | không có |
| SIR / SAR (BSS Eval) | không có |
| PESQ / STOI | không có |
| Mask IoU / F1 | không có |
| Accuracy | **đã xoá** cùng classifier |

Nói cách khác, repo **không có metric đánh giá chất lượng tách nguồn**. Loss L1 trên mask là
proxy: nó đo mức khớp của mask, không đo chất lượng nghe được của waveform. Một model có L1
thấp vẫn có thể tách nghe dở nếu mask đúng nhưng pha sai.

### 11.4 Muốn thêm metric tách nguồn thì cần

1. **Sinh cặp (mixture, nguồn thật) tất định.** `__getitem__` hiện lấy mẫu ngẫu nhiên nên
   mỗi lần gọi cho một mix khác. Cần một chế độ val tất định (seed cố định theo index) để
   metric so sánh được giữa các lần chạy.
2. **Lưu `audios[n]` (nguồn sạch) trước khi trộn.** Dataset hiện **không trả** waveform nguồn
   sạch; chỉ trả phổ. Không có nó thì không tính được SDR.
3. **Tái tạo cả nguồn dự đoán lẫn nguồn thật** rồi so ở miền thời gian (hoặc dùng BSS Eval
   trên phổ). Xem công thức ở mục 10.2.
4. Với `num_per_mix > 2`, dùng BSS Eval thay vì SDR đôi một, vì SDR chỉ định nghĩa cho 2 nguồn.

---

## 12. Kiểm chứng thực tế

Đã chạy end-to-end trên dữ liệu tổng hợp dựng theo đúng định dạng `ducanh_audioVisual_dataset`
(20 video × 2 clip; `yolo_top_detections/*.npy`, `reshape_11025/*.wav`, `frame/<clip>/*.png`),
CPU, `--gpu_ids -1`.

| Kiểm tra | Kết quả |
|---|---|
| Chạy đủ `--tensorboard --measure_time` | pass; `data/coseparation_loss` `[(2,0.153),(4,0.1472)]`, `data/val_coseparation_loss` `[(2,0.1445),(4,0.139)]` |
| Sinh split theo video | `20 videos (4 held out) \| {'train.txt': 32, 'val.txt': 8}` — không có video nào nằm cả hai phía |
| Loss giảm | train 2.95 → 2.83; val 2.95 → 2.83 |
| Trực quan hoá | `mix/gt/pred.wav` (11025 Hz, 65535 mẫu) + 4 ảnh jpg + `weight.jpg` + `index.html` |
| Trọng số mô hình | `net_visual` `(4,3,224,224)→(4,512,1,1)`; UNet 5 và 7 tầng đều trả `(4,1,256,256)` ∈ (0,1) |
| `--continue_train` | resume đúng `total_batches` và `best_err`; LR đã decay giữ nguyên; không ghi đè best bằng model tệ hơn |
| `--freeze_visual` | chỉ còn một nhóm LR; chỉ `unet_*.pth` được cập nhật |
| Thông báo lỗi sớm | `--model` sai, thiếu `train.txt`, `--continue_train` không có state → đều báo `ValueError`/`FileNotFoundError` rõ ràng |
| Mô phỏng `script.sh` (trừ đường dẫn/gpu) | pass, loss val ~0.14 |

---

## 13. Nợ kỹ thuật và việc cần làm

Theo mức ưu tiên.

| # | Vấn đề | Vị trí | Mức độ |
|---|---|---|---|
| 1 | Không có `test.py`/inference, không có metric tách nguồn (SDR/SI-SDR) | toàn repo | **Thiếu chức năng** |
| 2 | Dataset không trả waveform nguồn sạch → không thể tính metric ở mục 11.4 | `data/ducanh_audioVisual_dataset.py` | Chặn việc thêm metric |
| 3 | Validation lấy mẫu ngẫu nhiên mỗi lần → metric không so sánh được giữa các run | `__getitem__` | Chặn việc thêm metric |
| 4 | Tái tạo dùng pha của hỗn hợp → chặn trên chất lượng tách | `save_visualization`, tương lai `test.py` | Giới hạn thiết kế |
| 5 | `--num_batch` mặc định 30000, dễ chạy nhầm rất lâu khi test nhanh | `options/train_options.py` | Dễ sai |
| 6 | ~10 CLI option đã chết (`classifier_pool`, `audio_pool`, `lr_classifier`, `lr_vocal_attributes`, `lr_facial_attributes`, `weights_classifier`, `weights_vocal`, `weights_facial`, `classifier_loss_weight`, `crossmodal_loss_weight`, `identity_feature_dim`, `triplet_loss_type`, `margin`, `mask_thresh`, `num_object_per_video`) | `options/*.py` | Gây nhầm lẫn |
| 7 | `models/networks.py` 803 dòng, phần lớn code chết; `*_Audio` có `NameError` ẩn | `models/networks.py` | Bảo trì |
| 8 | `dataset/` chứa dữ liệu mẫu của tác giả, dễ tưởng là dữ liệu thật | `dataset/` | Gây nhầm lẫn |
| 9 | `checkpoints_dir` mặc định là đường dẫn tuyệt đối của máy tác giả | `options/base_options.py` | Portability |
| 10 | Hai biến thể dataset cũ vẫn nằm trong repo, chứa đường dẫn hard-code | `data/audioVisual_dataset.py`, `data/hai_audioVisual_dataset.py` | Bảo trì |
| 11 | `opt.hdf5_path` thực chất là thư mục split — tên gây hiểu nhầm | `train.py:324` | Dễ sai |
| 12 | `model.zero_grad()` gọi trùng với `optimizer.zero_grad()` | `train.py:438,449` | Nhỏ |
| 13 | `grid_sample` thiếu `align_corners` → cảnh báo và lệch nhẹ phép unwarp | `train.py:148-150`, `models/audioVisual_model.py` | Nhỏ |
| 14 | `--tensorboard True/False` dùng `type=bool` → `"False"` vẫn thành `True` | `options/train_options.py` | Dễ sai |
| 15 | `models/models.py` import `torch` không dùng; `utils/utils.py` import `distutils.command.clean` (đã bị xoá khỏi Python 3.12+) | nhiều file | Nhỏ |

Về #14: `argparse` với `type=bool` biến bất kỳ chuỗi khác rỗng thành `True`. Muốn tắt thật
phải bỏ hẳn cờ, không viết `--tensorboard False`. Các cờ `type=bool` tương tự:
`validation_on`, `validation_visualization`, `subtract_mean`, `preserve_ratio`,
`enable_data_augmentation`, `log_freq`, `measure_time`.

Về #15: `from distutils.command.clean import clean` trong `utils/utils.py` hiện chạy được chỉ
vì `setuptools` cài shim cho `distutils`. Trên môi trường không có shim sẽ `ImportError` ngay
khi import `utils`, tức `train.py` không chạy được. Dòng này hoàn toàn không dùng tới.

---

## 14. Cách chạy

### 14.1 Trên server, như `script.sh`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --name run_av_2branch \
  --model audioVisual \
  --data_path /media/data4/home/lanlt/ducanh/dataset_av \
  --auto_split --val_ratio 0.1 \
  --batchSize 1 --nThreads 4 \
  --num_batch 100000 \
  --lr_steps 60000 90000 \
  --unet_num_layers 7 \
  --lr_unet 0.001 --lr_visual 0.0001 \
  --weighted_loss --optimizer sgd --mask_loss_type L1 \
  --coseparation_loss_weight 1 \
  --log_freq True --tensorboard True \
  --save_latest_freq 1000 \
  --validation_freq 200 --validation_batches 5 \
  --gpu_ids 0 \
  --checkpoints_dir /media/data4/home/lanlt/ducanh/final_2branch/checkpoint \
  --num_visualization_examples 4
```

`--data_path` phải chứa đồng thời ba thư mục con:

```
<data_path>/
├── yolo_top_detections/<bất kỳ>/<clip>.npy   # [frame_id, class, 0, 0, x1, y1, x2, y2]
├── reshape_11025/<bất kỳ>/<clip>.wav          # mono, 11025 Hz
└── frame/<bất kỳ>/<clip>/<frame_id:06d>.png   # ảnh RGB
```

`<bất kỳ>` là thư mục con tuỳ ý — code dùng `glob(..., recursive=True)` nên cấu trúc sâu bao
nhiêu cũng được, miễn tên thư mục gốc đúng ba tên trên. Tên file `.npy` phải chứa `_clip_`
theo dạng `<video>_clip_<số>` để `get_vid_name` tách đúng video.

### 14.2 Chạy tiếp khi bị ngắt

```bash
python train.py --name run_av_2branch --model audioVisual \
  --data_path /media/data4/home/lanlt/ducanh/dataset_av --auto_split \
  --num_batch 100000 --lr_steps 60000 90000 --unet_num_layers 7 \
  --weighted_loss --optimizer sgd --mask_loss_type L1 \
  --coseparation_loss_weight 1 --gpu_ids 0 \
  --checkpoints_dir /media/data4/home/lanlt/ducanh/final_2branch/checkpoint \
  --continue_train
```

Điều kiện: `checkpoints_dir/name/training_state.pth` phải tồn tại, tức đã chạy qua ít nhất
một mốc `--save_latest_freq`. Các siêu tham số kiến trúc (`--unet_num_layers`, `--weighted_loss`)
phải **giống hệt** lần chạy trước, nếu không `load_state_dict` sẽ lỗi shape mismatch.

### 14.3 Test nhanh trên CPU

```bash
python train.py --name smoke --model audioVisual --data_path <data_path> \
  --auto_split --batchSize 1 --nThreads 0 --num_batch 4 \
  --unet_num_layers 5 --optimizer sgd --freeze_visual \
  --gpu_ids -1 --checkpoints_dir ./checkpoints
```

Bắt buộc đặt `--num_batch` nhỏ, nếu không sẽ chạy 30000 batch. `--gpu_ids -1` là cách duy
nhất để chọn CPU.

### 14.4 Xem TensorBoard

```bash
tensorboard --logdir runs
```

Log ghi vào `runs/` **trong thư mục hiện hành**, không phải trong `checkpoints_dir`.
