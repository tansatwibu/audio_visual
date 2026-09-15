# PROJECT_ANALYSIS — Audio-Visual Co-Separation

Phân tích kỹ thuật của repo `audio_visual`: kiến trúc, luồng dữ liệu, kích thước tensor,
loss, optimizer, vòng lặp huấn luyện, validation, checkpoint, inference và cách tính metric.

> Mọi nhận định dưới đây được rút ra trực tiếp từ source code trong repo.
> Những chỗ suy luận (do thiếu file dữ liệu/tham số runtime) đều được ghi rõ là **suy luận**.

---

## 1. Tổng quan repo

| File | Vai trò |
|---|---|
| `train.py` (567 dòng) | Entry point duy nhất. Build model, optimizer, loss, vòng lặp train + validation + checkpoint. |
| `options/base_options.py` | Tham số chung (data path, audio STFT, batch size, gpu). |
| `options/train_options.py` | Tham số train (loss weight, LR, validation, unet, optimizer). |
| `options/test_options.py` | Tham số cho chế độ test. **Chưa có script test nào dùng tới.** |
| `models/models.py` | `ModelBuilder` — factory tạo 5 subnet. |
| `models/networks.py` | `Resnet18`, các biến thể UNet (`AudioVisual7layerUNet`, `AudioVisual5layerUNet`, `AudioPose7layerUNet`, `AudioVisualPose7layerUNet`), `weights_init`. |
| `models/audioVisual_model.py` | `AudioVisualModel` — lớp ghép toàn bộ pipeline forward. |
| `models/criterion.py` | `L1Loss`, `L2Loss`, `BCELoss`, `CELoss`, `TripletLoss`, `TripletLossCosine`. |
| `data/audioVisual_dataset.py` | Dataset đang dùng (theo `opt.model = 'audioVisualMUSIC'`). Có pose feature. |
| `data/ducanh_audioVisual_dataset.py` | Biến thể dataset (không có pose). |
| `data/hai_audioVisual_dataset.py` | Biến thể dataset khác (không có pose). |
| `data/custom_dataset_data_loader.py` | `DataLoader` với `collate_fn = object_collate`. |
| `utils/utils.py` | `warpgrid`, `magnitude2heatmap`, `istft_reconstruction`, `object_collate`, hàm mix audio theo SNR. |
| `utils/viz.py` | `HTMLVisualizer` — xuất bảng HTML có ảnh + audio. |
| `dataset/` | Các file `.h5` chứa **đường dẫn** `.npy` (không chứa ảnh/audio thật) + các file `.txt` danh sách clip. |

Điểm quan trọng: repo **không có `test.py` / `inference.py` / `metrics.py`**. Toàn bộ inference
và metric đều được mô tả ở mức "có thể suy ra từ forward pass" — xem mục 10 và 11.

Chạy: `python train.py [options]`. `BaseOptions.parse()` tự động in ra toàn bộ option và ghi
`opt.txt` vào `checkpoints_dir/name/`.

---

## 2. Architecture

### 2.1 Sơ đồ tổng thể

```
                     ┌──────────────────────────────┐
   visuals  ────────►│ net_visual  (Resnet18)       │──► visual_feature
   (B,3,224,224)     │ maxpool, no FC               │    (B,512,1,1)
                     └──────────────────────────────┘
                                                          (không dùng cho loss,
                                                           chỉ để tham chiếu)
   audio_mags ──────►│ net_vocal (Resnet18+FC)      │──► audio_embeddings_gt
   (B,1,255,256)     │ maxpool, fc 512→64           │    (B,64)
                     └──────────────────────────────┘
                                                          (triplet loss đã bị comment)

   audio_mix_mags ──► log ──────►┌──────────────────────┐
   (B,1,256,256)                 │ net_unet             │──► mask_prediction
   visual_feature ──────────────►│ (UNet 7 tầng, ngf=64)│    (B,1,256,256)
   poses ───────────────────────►└──────────────────────┘
                                                          │
   audio_mix_mags ────────────────────────────────────────┴──► separated_spectrogram
                                                                  (B,1,256,256)
                                                          │
                                        log(·) ───────────┴──► ┌───────────────────┐
                                                               │ net_classifier    │──► label_prediction
                                                               │ (Resnet18 + FC)   │    (B, num_classes)
                                                               └───────────────────┘
```

### 2.2 Các subnet

`AudioVisualModel.__init__` nhận tuple 5 net:
`(net_visual, net_unet, net_classifier, net_vocal, net_identity)`.

| Net | Class (networks.py) | Input | Output | Ghi chú |
|---|---|---|---|---|
| `net_visual` | `Resnet18(pool_type='maxpool', with_fc=False)` | `(B,3,224,224)` | `(B,512,1,1)` | `conv1` được thay để nhận số channel tuỳ biến; pretrained ResNet18. `fc_out=256` truyền vào nhưng **không có tác dụng** vì `with_fc=False`. |
| `net_unet` | `AudioVisual7layerUNet` (do `models.py` trả về khi `unet_num_layers=7`) | `x=(B,1,256,256)`, `visual_feat=(B,512,1,1)` | `(B,1,256,256)`, sigmoid | `unet_ngf=64`. Xem cảnh báo 2.4. |
| `net_classifier` | `Resnet18(pool_type='maxpool', with_fc=True, fc_in=512, fc_out=num_classes)` | `(B,1,256,256)` | `(B,num_classes)` | `number_of_classes` mặc định 15. |
| `net_vocal` | `Resnet18(pool_type='maxpool', with_fc=True, fc_in=512, fc_out=64)` | `(B,1,255,256)` | `(B,64)` | `identity_feature_dim=64`. |
| `net_facial_attribtes` | `build_facial(...)` (trong `models.py` bị comment, class `AudioPose7layerUNet` không dùng) | — | — | **Được build, đưa vào optimizer và được save checkpoint, nhưng KHÔNG xuất hiện trong `forward`** → nhánh chết. |

### 2.3 UNet 7 tầng — chi tiết tầng (ngf=64, input 256×256)

`unet_conv` = `Conv2d(k=4, s=2, p=1) → BatchNorm2d → LeakyReLU(0.2)`.
`unet_upconv` = `ConvTranspose2d(k=4, s=2, p=1) → BatchNorm2d → ReLU` (tầng ngoài cùng thay BN+ReLU bằng `Sigmoid`).

Encoder (channel, H, W):

| Tầng | Out channels | Kích thước |
|---|---|---|
| conv1 | 64 | 128×128 |
| conv2 | 128 | 64×64 |
| conv3 | 256 | 32×32 |
| conv4 | 512 | 16×16 |
| conv5 | 512 | 8×8 |
| conv6 | 512 | 4×4 |
| conv7 | 512 | 2×2 |

Bottleneck: `visual_feat` được `repeat` lên `(B,512,2,2)` rồi `cat` với `conv7` theo channel → `(B,1024,2,2)`.

Decoder:

| Tầng | Vào | Ra | Skip |
|---|---|---|---|
| upconv1 | 1024 | 512 @ 4×4 | + conv6 → 1024 |
| upconv2 | 1024 | 512 @ 8×8 | + conv5 → 1024 |
| upconv3 | 1024 | 512 @ 16×16 | + conv4 → 1024 |
| upconv4 | 1024 | 256 @ 32×32 | + conv3 → 512 |
| upconv5 | 512 | 128 @ 64×64 | + conv2 → 256 |
| upconv6 | 256 | 64 @ 128×128 | + conv1 → 128 |
| upconv7 | 128 | 1 @ 256×256 | Sigmoid → mask ∈ (0,1) |

`AudioVisual7layerUNet.__init__` nhận `visual_dim=512` và dùng nó cho `upconv1` input channel,
nên nếu đổi chiều visual feature phải truyền đúng `visual_dim`.

### 2.4 ⚠️ Bất nhất kiến trúc (quan trọng)

`models.py::build_unet` với `unet_num_layers=7` trả về **`AudioVisual7layerUNet`**, có signature:

```python
def forward(self, x, visual_feat):        # 2 tham số
```

nhưng `AudioVisualModel.forward` lại gọi:

```python
mask_prediction = self.net_unet(audio_log_mags, visual_feature, pose_feature)   # 3 tham số
```

⇒ **Chạy `train.py` với code hiện tại sẽ lỗi `TypeError`** (hoặc lỗi thiếu key `'poses'` tuỳ
dataset) trước cả khi vào vòng lặp train. Biến thể đúng cho thiết kế này là
`AudioPose7layerUNet` (đã import trong `models.py` nhưng không được dùng), với:

```python
def forward(self, x, visual_feature, pose_feat)
```

Ngoài ra, biến thể `AudioVisualPose7layerUNet` (đã comment toàn bộ) là phiên bản fusion
visual+pose kiểu khác:

- `AudioPose7layerUNet`: bỏ qua `visual_feature`, chỉ dùng pose.
  `pose_feat.view(B, 256, 20, 68)` → conv 256→128→64 → maxpool 2×2 → 10×34 →
  flatten `64*340 = 21760` → `Linear(21760, 512)`.
- `AudioVisualPose7layerUNet`: ghép visual vào pose.
  `visual_feature.view(B,1,1,256).repeat(1,68,20,1)` và `pose_feat.view(B,68,20,256)`,
  `cat(dim=3)` → `(B,68,20,512)` → `view(B,512,20,68)` → conv 512→128→64 → maxpool → `Linear(128*17*10=21760, 512)`.

Cả hai đều **giả định pose feature có 256×20×68 = 348.160 phần tử mỗi mẫu**.

### 2.5 Luồng feature pose trong dataset đang dùng

`data/audioVisual_dataset.py` đọc pose từ hard-code:

```python
pose_path = "/media/data4/home/lanlt/Datle/LAGCN/feat"
pose_clip = os.path.join(pose_path, get_clip_name(clip_det_paths[n]) + ".npy")
pose_feat = np.load(pose_clip); pose_feat = torch.squeeze(...).unsqueeze(0)
```

Và đường dẫn frame/audio cũng hard-code:

```python
def get_frame_root(npy_path):  # /home/lanlt/Datle/yolo_ -> /media/data5/users/dalt/, top_detections_new -> frame_new
def get_audio_root(npy_path):  # yolo_ -> '', top_detections_new -> reshape_11025_new
```

Đây là **rào cản portability nghiêm trọng**: 3 biến thể dataset hard-code các tiền tố đường dẫn
khác nhau, không có tham số `--data_path` nào thực sự được dùng để build đường dẫn frame/audio
(`opt.data_path` chỉ được khai báo trong `base_options.py`).

---

## 3. Data flow

### 3.1 Từ file h5 → danh sách clip

`AudioVisualMUSICDataset.initialize`:

1. Đọc `os.path.join(opt.hdf5_path, opt.mode + '_yolo_solo.txt')` (ví dụ `train_yolo_solo.txt`).
   Mỗi dòng là đường dẫn tuyệt đối tới file `.npy` chứa bounding boxes.
   *(`dataset/*.h5` có key `detection`/`image` cũng chứa danh sách `.npy`, nhưng code lại đọc `.txt`;
   phần `h5py.File` đã bị comment.)*
2. Gom clip theo video id: `get_vid_name()` = 11 ký tự đầu của tên file. Mục đích: **không trộn 2 clip
   từ cùng một video** khi mix (đảm bảo hai nguồn âm thanh độc lập).
3. `transforms`: `val` → `Resize((224,224))`; `train` → `Resize((256,256))` + `RandomCrop(224)`
   (hoặc `Resize(256)+RandomCrop(224)` nếu `preserve_ratio`), cộng `Normalize(ImageNet mean/std)`.
4. Nếu `with_additional_scene_image`: nạp `opt.scene_path` (ví dụ `dataset/ADE.h5`) với key `image`.

### 3.2 Một mẫu `__getitem__`

```
random.sample(video_ids, NUM_PER_MIX=2)          # chọn 2 video
for n in 0..NUM_PER_MIX-1:
    clip_path[n] = random.choice(detection_dic[video[n]])
    bbs[n]       = sample_object_detections(load(clip_path[n]))   # 1 detection / class
```

`sample_object_detections` nhóm detection theo class id rồi lấy **ngẫu nhiên 1 detection cho mỗi class**
→ với dataset 2 class, mỗi clip đóng góp ~2 object.

Với mỗi clip `n`:

1. `vid = random.randint(1, 1e11)` — id duy nhất đánh dấu clip nguồn (không dùng tên video thật).
2. `librosa.load(wav, sr=11025)` → `sample_audio(window=65535)` (tile lặp nếu audio ngắn hơn window;
   cửa sổ lấy ngẫu nhiên bằng `randrange`).
3. `augment_audio`: nhân biên độ với `random()+0.5` ∈ [0.5, 1.5] rồi clip về [-1, 1]. (Bản
   `audiomentations` bị comment.)
4. `generate_spectrogram_magphase(audio, n_fft=1022, hop=256)` → `(mag, phase)`, mỗi cái `(1,512,256)`.
5. Với **mỗi detection** `i`:
   - nạp pose `.npy` → `objects_poses`
   - `Image.open(frame).crop(bbox)` → `augment_image` (flip ngang p=0.5, đổi brightness/color)
     → `vision_transform` → `(3,224,224)`
   - `label = bbox_class - 1` (class bắt đầu từ 0)
   - **copy spectrogram của clip cho object này** (`objects_audio_mag`, `objects_audio_phase`)
   - `objects_vids.append(vid)`
6. Nếu `with_additional_scene_image`: thêm 1 ảnh scene với `label = number_of_classes - 1`.

Sau khi có đủ object của 2 clip:

```
audio_mix = np.asarray(audios).sum(axis=0) / NUM_PER_MIX      # trộn = trung bình cộng biên độ
audio_mix_mag, audio_mix_phase = stft(audio_mix)
```

rồi **copy spectrogram trộn cho mọi object**. Cuối cùng `np.vstack` tất cả list và trả về dict:

```python
{'labels', 'audio_mags', 'audio_mix_mags', 'vids', 'visuals', 'poses'}
# + {'audio_phases','audio_mix_phases'} chỉ khi mode ∈ {val, test}
```

**Hệ quả thiết kế:** batch là *tập object*, không phải tập clip. Nhiều hàng trong cùng batch
chia sẻ `vid` và chia sẻ `audio_mix_mags`; mỗi hàng có `audio_mags` riêng (nguồn của object đó).
Đây chính là thứ mà co-separation loss (mục 5.2) khai thác.

### 3.3 Collate

`CustomDatasetDataLoader` dùng `collate_fn=object_collate` (`utils/utils.py`). Khác với collate mặc định,
với mảng numpy nó dùng `torch.cat(..., 0)` thay vì `stack`:

```python
if elem_type.__name__ == 'ndarray':
    return torch.cat([torch.from_numpy(b) for b in batch], 0)   # concat dù dimension khác nhau
```

→ dimension thứ 0 của tensor batch = **tổng số object trong batch**, không bằng `batchSize`.
`batchSize=32` thực chất là số *clip mix* (mỗi phần tử dataset là 1 lần mix 2 clip), nên batch
thực tế lớn hơn 32 đáng kể.

`DataLoader` đặt `shuffle=False` cho cả train và val, `num_workers=opt.nThreads` (train) và `2` (val).
Việc random đến từ chính `__getitem__` + `__len__ = batchSize*num_batch`, nên không cần shuffle.

---

## 4. Tensor dimensions

Ký hiệu: `B` = số object trong batch (≈ `batchSize × NUM_PER_MIX × objects_per_clip`),
`C` = số class (`number_of_classes`, mặc định 15).

### 4.1 Tham số âm thanh (mặc định)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `audio_sampling_rate` | 11025 | `base_options.py` |
| `audio_window` | 65535 | `base_options.py` |
| `stft_frame` (n_fft) | 1022 | `base_options.py` |
| `stft_hop` | 256 | `base_options.py` |
| số bin tần số | `1022//2 + 1 = 512` | suy ra |
| số frame thời gian | `1 + 65535//256 = 256` | suy ra (center=True) |

### 4.2 Bảng tensor xuyên suốt

| Bước | Tensor | Shape (trước) | Shape (sau) | Ghi chú |
|---|---|---|---|---|
| STFT magnitude | `audio_mag` / `audio_mix_mag` | waveform `(65535,)` | `(1,512,256)` | mỗi object 1 bản |
| Batch audio | `audio_mags`, `audio_mix_mags` | list `(1,512,256)` | `(B,1,512,256)` | sau `object_collate` |
| Warp log-freq | `audio_mix_mags`, `audio_mags` | `(B,1,512,256)` | `(B,1,256,256)` | `warpgrid(B,256,T=256, warp=True)` + `F.grid_sample` |
| GT mask | `gt_masks` | `(B,1,256,256)` | `(B,1,256,256)` | `audio_mags / audio_mix_mags`, `clamp_(0,5)` |
| Input UNet | `audio_log_mags` | `(B,1,256,256)` | — | `log(audio_mix_mags).detach()` |
| Visual feature | `visual_feature` | `(B,3,224,224)` | `(B,512,1,1)` | ResNet18 maxpool, xem 4.3 |
| Pose feature | `pose_feature` | `(B,348160)` | `(B,256,20,68)` | `view` trong `AudioPose7layerUNet` |
| Mask | `mask_prediction` | `(B,1,256,256)` | `(B,1,256,256)` | sigmoid, ∈ (0,1) |
| Separation | `separated_spectrogram` | `(B,1,256,256)` | — | `audio_mix_mags * mask_prediction` |
| Log-spec cho classifier | `spectrogram2classify` | `(B,1,256,256)` | — | `log(sep + 1e-10)` |
| Classifier output | `label_prediction` | `(B,1,256,256)` | `(B,15)` | ResNet18 + FC |
| Loss weight | `weight` | `(B,1,256,256)` | — | `clamp(log1p(audio_mix_mags), 1e-3, 10)` |
| Audio embedding (GT) | `audio_embeddings_gt` | `(B,1,255,256)` | `(B,64)` | `audio_mags[:,:,:-1,:]` (bỏ frame cuối) |
| Audio embedding (pred) | `audio_embeddings_pred` | `(B,1,255,256)` | `(B,64)` | hiện **không dùng để tính loss** |
| Label GT | `gt_label` | `(B,1)` | `(B,)` long | `.squeeze(1).long()` |
| Video id | `vids` | list | `(B,1)` | dùng để gom theo clip |

### 4.3 Vì sao `visual_feature` là `(B,512,1,1)`

`build_visual` được gọi với `fc_out=256` nhưng không truyền `with_fc=True`, nên
`Resnet18.__init__(with_fc=False)`, và nhánh `forward` chạy:

```python
x = F.adaptive_max_pool2d(x, 1)          # (B,512,1,1)
return x.view(x.size(0), -1, 1, 1)       # (B,512,1,1)
```

Trong `AudioVisual7layerUNet.forward`, `visual_dim=512` khớp với shape này. Nếu sau này bật
`visual_pool='conv1x1'`, `build_visual` sẽ dùng `fc_in=6272` và `fc_out=256`, lúc đó
`visual_dim` của UNet phải được set lại tương ứng (`AudioVisual7layerUNet` hiện hard-code 512).

### 4.4 Lưu ý về classifier và kích thước đầu vào

`net_classifier` = ResNet18 với `input_channel=1` (lấy từ `opt.unet_output_nc`) nhận
`(B,1,256,256)`. ResNet18 nguyên bản thiết kế cho 224×224; ở đây 256×256 vẫn chạy được
do dùng `adaptive_*_pool`, nhưng chi phí tính toán tăng ~30%.

---

## 5. Loss

### 5.1 Classification loss

```python
loss_classification = criterion.CELoss()
classifier_loss = loss_classification(output['pred_label'],
                                      Variable(output['gt_label'], requires_grad=False)) \
                  * opt.classifier_loss_weight          # default 1
```

`CELoss._forward` = `F.cross_entropy(pred, target)`, tức softmax + NLL.
`pred_label` là **logits** `(B,15)`, `gt_label` là `(B,)` long.

### 5.2 Co-separation loss (`get_coseparation_loss` trong `train.py`)

Đây là loss lõi của bài toán. Thuật toán:

1. Lấy `vids = output['vids'].squeeze(1).cpu().numpy()`, xây `vid_index_dic` ánh xạ
   video id → chỉ số 0..K-1 (K = số clip khác nhau trong batch, thường = `NUM_PER_MIX` × số mix).
2. Với mỗi object `i`, **cộng dồn** mask dự đoán vào `predicted_mask_list[video của i]`:

```python
predicted_mask_list[vid_index_dic[vids[i]]] += mask_prediction[i,:,:,:]
```

   Ý tưởng: mỗi clip có nhiều object (nhiều nhạc cụ cùng phát ra trong clip đó); các object
   này **cùng nguồn âm thanh**, nên tổng mask của chúng phải xấp xỉ mask lý tưởng của clip.
3. `gt_mask_list` và `weight_list` lấy **object đầu tiên** của mỗi clip (mọi object cùng clip có
   `gt_mask` và `weight` giống nhau do dataset copy).
4. Nếu `mask_loss_type == 'BCE'`: `clamp(predicted_mask, 0, 1)` trước khi tính loss.
5. `loss_coseparation(predicted_mask_list, gt_mask_list, weight_list)`.

`BaseLoss.forward` với input dạng list:

```python
errs = [self._forward(preds[n], targets[n], weight[n]) for n in range(N)]
err = torch.mean(torch.stack(errs))     # trung bình trên các clip
```

Với `L1Loss`: `mean(weight * |pred - target|)`. Với `L2Loss`: `mean(weight * (pred-target)^2)`.
Với `BCELoss`: `binary_cross_entropy(pred, target, weight=weight)`.

Vì `gt_masks` là **ratio mask** (`audio_mags/audio_mix_mags`, clamp 0..5), `predicted_mask` là
tổng các mask trong một clip, nên về lý tưởng tổng này = ratio mask của clip đó.

Nhân với `opt.coseparation_loss_weight` (default **20**).

### 5.3 Triplet / cross-modal loss — đã bị vô hiệu hoá

`loss_triplet` vẫn được khởi tạo, `.cuda()`, và đặt trong `crit`, nhưng **toàn bộ** logic
`get_crossmodal_loss` (2 phiên bản) đã bị comment out, cùng với việc backward loss này.
`output['visual_embadding']`, `output['audio_embeddings_gt']`, `output['audio_embeddings_pred']`
vì thế **không đóng góp gradient nào**.

Hệ quả thực tế:
- `net_vocal` (và `net_facial_attribtes`) **không nhận gradient** trong cấu hình hiện tại.
- Chúng vẫn nằm trong optimizer và vẫn được lưu checkpoint → file `.pth` `vocal_*`/`facial_*`
  hầu như chỉ chứa trọng số pretrained/khởi tạo.

### 5.4 Tổng loss trong train

```python
classifier_loss.backward(retain_graph=True)
coseparation_loss.backward(retain_graph=True)
optimizer.step()
```

Hai loss được backward **riêng rẽ** (không cộng thành một scalar). Không có `loss_total`.
`retain_graph=True` ở lần backward thứ nhất để graph còn dùng cho lần thứ hai.
`model.zero_grad()` được gọi trước forward và `optimizer.zero_grad()` trước backward — trùng lặp
nhưng vô hại.

---

## 6. Optimizer

```python
def create_optimizer(nets, opt):
    param_groups = [{'params': net_visual.parameters(),     'lr': opt.lr_visual},
                    {'params': net_unet.parameters(),       'lr': opt.lr_unet},
                    {'params': net_classifier.parameters(), 'lr': opt.lr_classifier},
                    {'params': net_vocal.parameters(),      'lr': opt.lr_vocal_attributes},
                    {'params': net_facial_attribtes.parameters(), 'lr': opt.lr_facial_attributes}]
    if opt.optimizer == 'sgd':   torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
```

| Tham số | Default | Ý nghĩa |
|---|---|---|
| `lr_visual` | 1e-4 | LR nhánh visual (pretrained, LR thấp) |
| `lr_unet` | 1e-3 | LR nhánh mask UNet |
| `lr_classifier` | 1e-3 | LR nhánh phân loại |
| `lr_vocal_attributes` | 1e-3 | LR nhánh audio embedding (**không có gradient**) |
| `lr_facial_attributes` | 1e-4 | LR nhánh facial (**không dùng**) |
| `optimizer` | `sgd` | `sgd` hoặc `adam` |
| `beta1` | 0.9 | momentum (SGD) / beta1 (Adam) |
| `weight_decay` | 1e-4 | L2 regularizer |
| `lr_steps` | `[10000, 20000]` | mốc batch để giảm LR |
| `decay_factor` | 0.1 | hệ số giảm LR |

**LR scheduling thủ công** (không dùng `torch.optim.lr_scheduler`):

```python
def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

if total_batches in opt.lr_steps:
    decrease_learning_rate(optimizer, opt.decay_factor)
```

Không có warmup, không có gradient clipping. `opt.continue_train` được khai báo trong
`train_options.py` nhưng **chưa bao giờ được đọc** → không có resume thực sự (xem mục 9).

Thiết bị: `opt.device = torch.device("cuda")` hard-code trong `train.py`; model bọc
`torch.nn.DataParallel(model, device_ids=opt.gpu_ids)` với `gpu_ids` mặc định `'1'`.
`base_options.parse()` gọi `torch.cuda.set_device(gpu_ids[0])`.

---

## 7. Training loop

Cấu trúc (`train.py`, phần cuối file):

```python
for epoch in range(1 + opt.epoch_count, opt.niter + 1):        # niter=1
    for i, data in enumerate(dataset):                          # dataset = CustomDatasetDataLoader
```

Vì `__len__` của dataset = `batchSize * num_batch` và `niter=1`, thực chất đây là
**một epoch với `num_batch` (default 30000) bước**, mỗi bước là một batch mix ngẫu nhiên.
Số `epoch` chỉ có ý nghĩa hình thức. `num_batch=30000`, `batchSize=32` → 960.000 lần mix.

Mỗi iteration (`total_batches` tăng đơn điệu, là biến "global step" thực sự):

1. **Đo thời gian** (nếu `--measure_time`): `torch.cuda.synchronize()` ở các mốc
   `data_loaded → forwarded → backwarded`, tích luỹ vào `data_loading_time`,
   `model_forward_time`, `model_backward_time`.
2. **Forward**: `model.zero_grad()` → `output = model.forward(data)`.
   Lưu ý `model` là `DataParallel`, nên `model.forward` gọi thẳng `Module.forward` (bỏ hook
   phân tán) — chấp nhận được nhưng không phải cách gọi thông thường (`model(data)`).
3. **Tính loss**: `classifier_loss`, `coseparation_loss` (mục 5).
4. **Log batch loss** vào `batch_classifier_loss`, `batch_coseparation_loss`.
5. **Backward**: `optimizer.zero_grad()` → `classifier_loss.backward(retain_graph=True)` →
   `coseparation_loss.backward(retain_graph=True)` → `optimizer.step()`.
6. **Display** mỗi `display_freq=10` batch: in loss trung bình, time trung bình, rồi reset list.
   Nếu `--tensorboard` (mặc định `False`) ghi `data/classifier_loss`, `data/coseparation_loss`.
7. **Save latest** mỗi `save_latest_freq=200` batch (mục 9).
8. **Validation** mỗi `validation_freq=500` batch nếu `validation_on` (mục 8).
9. **Giảm LR** khi `total_batches ∈ lr_steps`.

Sau khi hết epoch: `opt.mode = 'train'` (đặt lại sau khi validation đổi mode).

Lưu ý về `opt.mode`: dataset đọc `opt.mode` **tại thời điểm `__getitem__`** để quyết định
augmentation và có trả về `audio_phases` hay không. Khi validation chạy, code set
`opt.mode = 'val'` rồi set về `'main'` — nhưng `'main'` **không phải** giá trị mà dataset
so sánh (`'train'`, `'val'`, `'test'`). Vì `train` dataset đã được tạo trước đó và chỉ
kiểm tra `mode == 'val'` cho phase nên vẫn hoạt động, song đây là chỗ dễ phát sinh bug.

---

## 8. Validation

`display_val(model, crit, writer, index, dataset_val, opt)`:

1. Tạo `save_dir = ./{checkpoints_dir}/{name}/visualization`.
2. Khởi tạo `HTMLVisualizer` với header
   `['Filename', 'Input Mixed Audio', 'Predicted Audio' 'GroundTruth Audio', 'Predicted Mask', 'GroundTruth Mask', 'Loss weighting']`
   (chuỗi `'Predicted Audio' 'GroundTruth Audio'` bị **thiếu dấu phẩy** → tự nối thành 1 cột).
3. Trong `torch.no_grad()`, duyệt `dataset_val`; với `i < validation_batches` (default 10):
   - `output = model.forward(val_data)`
   - `classifier_loss = CE(...) * classifier_loss_weight`
   - `coseparation_loss = get_coseparation_loss(...) * coseparation_loss_weight`
   - `accuracy = sum(gt_label == argmax(pred_label)) / B`
4. Batch thứ `i == validation_batches` (nếu còn) dùng **chỉ để visualize**
   (`opt.validation_visualization`) rồi `break`.
5. Tính trung bình, ghi TensorBoard (`data/val_classifier_loss`, `data/val_accuracy`,
   `data/val_coseparation_loss`), in ra console.
6. `return avg_coseparation_loss + avg_classifier_loss`.

Giá trị trả về này là `val_err` trong `train.py`, dùng để quyết định checkpoint `best`.

**Vấn đề trong `save_visualization`** (chỉ chạy khi `validation_visualization=True`):

- `_, pred_label = torch.max(output['pred_label'], 1)` → biến `output` **không tồn tại**
  (tên tham số là `outputs`) → `NameError`.
- `utils.istft_coseparation(...)` được gọi 4 lần nhưng `utils/utils.py` **chỉ có**
  `istft_reconstruction(mag, phase, hop_length, length)` → `AttributeError`.
- `label_list` trong `train.py` có **11 nhãn nhạc cụ** trong khi `number_of_classes` default là **15**
  → `IndexError` tiềm tàng khi visualize.
- Dùng `scipy.misc.imsave` — đã bị **loại bỏ từ SciPy 1.2** → cũng sẽ lỗi trên môi trường hiện đại.

⇒ Chức năng visualization thực tế đang hỏng; cần sửa trước khi dùng.

`dataset_val` được tạo bằng cách tạm đặt `opt.mode = 'val'`, gọi `CreateDataLoader(opt)` rồi trả
`opt.mode = 'train'`. `__len__` của val = `batchSize * validation_batches`.

---

## 9. Checkpoint

### 9.1 Ghi

Hai loại, đều lưu **state_dict của từng subnet riêng lẻ** (không lưu chung 1 file, không lưu optimizer):

```python
torch.save(net_visual.state_dict(),        .../'visual_latest.pth')
torch.save(net_unet.state_dict(),          .../'unet_latest.pth')
torch.save(net_classifier.state_dict(),    .../'classifier_latest.pth')
torch.save(net_vocal.state_dict(),         .../'vocal_latest.pth')
torch.save(net_facial_attribtes.state_dict(), .../'facial_latest.pth')
```

- `*_latest.pth`: mỗi `save_latest_freq` (default 200) batch.
- `*_best.pth`: khi `val_err < best_err` (khởi tạo `best_err = float("inf")`), mỗi `validation_freq` batch.
  In `'saving the best model (epoch %d, total_batches %d) with validation error %.3f'`.

Đường dẫn: `os.path.join('.', opt.checkpoints_dir, opt.name, ...)`. `checkpoints_dir` mặc định là
một đường dẫn tuyệt đối hard-code của máy tác giả
(`/media/data4/home/lanlt/Datle/co-separation_dat_copy/checkpoint_14_4/`) → **phải override khi chạy**.

Cùng thư mục đó cũng chứa `opt.txt` do `BaseOptions.parse()` ghi.

### 9.2 Đọc (resume)

- `--continue_train` tồn tại trong `train_options.py` nhưng **không có code nào đọc nó**.
- `build_visual/build_unet/build_classifier/build_vocal/build_facial` hỗ trợ `weights=...` để
  `load_state_dict(torch.load(...))` (map_location `'cpu'` trong `models.py`).
  Riêng `build_vocal` và `build_facial` còn lọc bớt key không khớp shape (partial load).
- Vì **optimizer state / total_batches / best_err không được lưu**, resume thực sự là không thể
  (LR schedule và best-error tracking sẽ bị reset).

⇒ Kết luận: checkpoint chỉ đủ cho inference/eval, chưa đủ cho train-resume.

---

## 10. Inference

### 10.1 Hiện trạng

Repo **chỉ có `train.py`**. Không có script test/inference. `options/test_options.py` và
`data/*_dataset.py` với `mode='test'` cho thấy dự định thiết kế, nhưng phần thực thi còn thiếu
(`TestOptions` cũng thiếu hẳn các tham số như `unet_ngf`? — không, có đủ; nhưng thiếu
`--optimizer`, `--num_per_mix`, `--scene_path`… nên không chạy trực tiếp được).

### 10.2 Đường suy luận từ forward pass

Inference cho một cặp clip gồm:

1. Nạp weights: `--weights_visual`, `--weights_unet`, `--weights_classifier` (và `--weights_vocal`).
2. Với mỗi cửa sổ âm thanh trộn (`hop_size=0.05` trong `TestOptions` — trượt cửa sổ):
   - STFT hỗn hợp → magnitude `(1,512,256)`; warp log-freq → `(1,256,256)`.
   - Trích visual feature từ frame của từng nhạc cụ → `(N,512,1,1)`.
   - `net_unet(log(mix_mag), visual_feature, pose_feature)` → mask `(N,1,256,256)`.
   - Tổng mask theo clip → mask tách cho từng nguồn.
3. Tái tạo waveform: `mag_sep = mix_mag * mask`; dùng `utils.istft_reconstruction(mag, phase, hop_length, length=65535)`
   với **pha của hỗn hợp** (không ước lượng pha — "magonly" như `TestOptions.spectrogram_type`).
   Hàm này: `librosa.istft(mag*exp(1j*phase), hop_length, length)` rồi `np.clip(wav,-1,1)`.
4. Ghép các cửa sổ (overlap theo `hop_size`) thành waveform đầy đủ.
5. Trực quan hoá: `utils.magnitude2heatmap` (log10 + colormap JET) và `HTMLVisualizer`
   (bảng ảnh + thẻ `<audio>`/`<video>`).

**Lưu ý:** hàm `istft_coseparation` mà `train.py` gọi **không tồn tại**; hàm thật là
`istft_reconstruction` và nó không nhận `length` mặc định bằng `audio_window` (mặc định 65535 — tình cờ khớp).

### 10.3 Vấn đề pha

Vì chỉ magnitude được dự đoán (`unet_output_nc=1`, mask 1 kênh), pha phải lấy từ hỗn hợp.
Điều này giới hạn chất lượng tách nguồn (pha của hỗn hợp ≠ pha của từng nguồn).
`complex` trong `TestOptions.spectrogram_type` gợi ý một nhánh 2 kênh (real/imag) đã được
tính đến trong code comment (`pred_spec_A1_real/imag`) nhưng chưa hoạt động.

---

## 11. Metric calculation

### 11.1 Metric hiện có

Chỉ **một** metric phân loại, tính trong `display_val`:

```python
_, pred_label = torch.max(output['pred_label'], 1)
accuracy = torch.sum(gt_label == pred_label).item() / pred_label.shape[0]
```

→ **top-1 classification accuracy** trung bình trên các batch validation (không weighted, không top-k).

### 11.2 "Metric" dùng cho model selection

```python
val_err = avg_coseparation_loss + avg_classifier_loss   # cả hai đã nhân loss weight (20 và 1)
```

Đây là tổng loss validation có trọng số, **không phải metric chất lượng tách nguồn**. Vì
`coseparation_loss_weight=20` áp đảo, `best_err` thực chất gần như chỉ theo dõi mask loss.

### 11.3 Metric KHÔNG có (khoảng trống)

- **Không có SDR / SI-SDR / SIR / SAR** — tức không có đánh giá chuẩn của bài toán source separation.
- **Không có** đánh giá trên waveform tái tạo (không tính được vì pipeline eval chưa tồn tại).
- Trong `utils/utils.py` có `cal_adjusted_rms`, `cal_rms`, `mix`, `mix_2` — đây là các helper
  **trộn audio theo SNR cho mục đích tạo dữ liệu**, không phải metric đánh giá.
  (`mix_2` còn tham chiếu biến `clean_wav` không tồn tại → hàm này chưa chạy được.)
- TensorBoard chỉ log loss và accuracy.

### 11.4 Muốn thêm metric tách nguồn thì cần

1. `test.py` chạy sliding window (`TestOptions.hop_size`) → waveform từng nguồn.
2. Đối chiếu với waveform sạch (`reshape_11025/<clip>.wav`).
3. Cài đặt SI-SDR (đơn giản, không cần alignment) hoặc dùng `mir_eval.separation.bss_eval_sources`
   cho SDR/SIR/SAR chuẩn BSS-Eval.

---

## 12. Tổng hợp vấn đề cần xử lý (theo mức ưu tiên)

| # | Vấn đề | Vị trí | Mức độ |
|---|---|---|---|
| 1 | `ModelBuilder.build_unet` trả `AudioVisual7layerUNet` (2 tham số) nhưng model gọi với 3 tham số (audio, visual, pose) | `models/models.py` ↔ `models/audioVisual_model.py:72` | **Chặn chạy** |
| 2 | Đường dẫn pose/frame/audio hard-code tuyệt đối | `data/audioVisual_dataset.py` | **Chặn chạy** |
| 3 | `utils.istft_coseparation` không tồn tại (đúng là `istft_reconstruction`) | `train.py:89,102,104` | **Chặn** (khi bật visualization) |
| 4 | `save_visualization` dùng biến `output` thay vì `outputs` | `train.py:44` | **Chặn** (khi bật visualization) |
| 5 | `scipy.misc.imsave` đã bị xoá khỏi SciPy ≥ 1.2 | `train.py:3` | **Chặn** (khi bật visualization) |
| 6 | `label_list` 11 nhãn vs `number_of_classes=15` | `train.py:41` | Bug |
| 7 | `net_vocal` / `net_facial_attribtes` không nhận gradient (triplet loss bị comment) | `train.py`, `models/audioVisual_model.py` | Lãng phí tính toán, checkpoint vô nghĩa |
| 8 | `--continue_train` không được đọc; không lưu optimizer state | `train.py` | Thiếu chức năng |
| 9 | Không có `test.py`/inference, không có metric tách nguồn (SDR) | toàn repo | Thiếu chức năng |
| 10 | `checkpoints_dir` mặc định là đường dẫn tuyệt đối của máy tác giả | `options/base_options.py` | Portability |
| 11 | `torch.device("cuda")` hard-code, không hỗ trợ CPU/`-1` dù `--gpu_ids` có nhánh CPU | `train.py:385` | Portability |
| 12 | `opt.mode = 'main'` không khớp giá trị dataset hiểu | `train.py` | Bug tiềm ẩn |
| 13 | `DataParallel.forward` được gọi trực tiếp (`model.forward(data)`) | `train.py` | Bug tiềm ẩn (hook không chạy) |
| 14 | 3 biến thể dataset trùng lặp (`audioVisual_`, `ducanh_`, `hai_`) nhưng chỉ 1 được wiring | `data/` | Maintainability |

---

## 13. Cách chạy (sau khi sửa các vấn đề chặn)

```bash
python train.py \
  --hdf5_path  dataset/sample_hdf5 \
  --checkpoints_dir ./checkpoints \
  --name exp01 \
  --gpu_ids 0 \
  --batchSize 8 \
  --num_batch 30000 \
  --num_per_mix 2 \
  --number_of_classes 15 \
  --optimizer sgd \
  --mask_loss_type L1 \
  --coseparation_loss_weight 20 \
  --classifier_loss_weight 1