
import copy
import glob
import os
import time

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
from imageio.v2 import imwrite as imsave

from options.train_options import TrainOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from models import criterion
from utils import utils, viz
from utils.utils import object_collate


def load_dataset(opt):
    """Build the only dataset this entry point supports.

    One --data_path must hold frame/, reshape_11025/ and yolo_top_detections/.
    'audioVisual' is the name script.sh uses and 'audioVisualMUSIC' is the
    dataset class name; both select the same two-branch loader.
    """
    if opt.model not in ('audioVisual', 'audioVisualMUSIC'):
        raise ValueError("--model must be 'audioVisual' for the two-branch "
                         "audio+visual setup, got %r" % opt.model)
    from data.ducanh_audioVisual_dataset import AudioVisualMUSICDataset
    dataset = AudioVisualMUSICDataset()
    dataset.initialize(opt)
    print('dataset [%s] was created' % dataset.name())
    return dataset


def build_split_lists(data_path, split_dir, val_ratio=0.1, seed=0):
    """Derive train.txt / val.txt from the detection files on disk.

    Clips are grouped by video so two clips of the same performance never land
    on opposite sides of the split, which would leak the validation set. The
    grouping reuses the dataset's own get_vid_name so the split and the loader
    always agree on what a video is.
    """
    pattern = os.path.join(data_path, 'yolo_top_detections', '**', '*.npy')
    npy_paths = sorted(glob.glob(pattern, recursive=True))
    if not npy_paths:
        raise FileNotFoundError('No detections matched %s' % pattern)

    from data.ducanh_audioVisual_dataset import get_vid_name
    by_video = {}
    for path in npy_paths:
        by_video.setdefault(get_vid_name(path), []).append(path)

    if len(by_video) < 2:
        raise ValueError('--auto_split needs at least 2 videos to hold one out, found %d'
                         % len(by_video))

    rng = np.random.RandomState(seed)
    videos = sorted(by_video)
    rng.shuffle(videos)

    n_val = max(1, min(len(videos) - 1, int(round(len(videos) * val_ratio))))
    val_videos = set(videos[:n_val])

    utils.mkdirs(split_dir)
    counts = {}
    for name, is_val in (('train.txt', False), ('val.txt', True)):
        paths = sorted(p for v in videos if (v in val_videos) is is_val
                       for p in by_video[v])
        with open(os.path.join(split_dir, name), 'w') as f:
            f.write('\n'.join(paths) + '\n')
        counts[name] = len(paths)

    return counts, len(videos), n_val


def create_loader(opt, num_workers):
    dataset = load_dataset(opt)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=object_collate)
    return dataset, loader


def create_optimizer(nets, opt):
    net_visual, net_unet = nets
    param_groups = [{'params': net_unet.parameters(), 'lr': opt.lr_unet}]
    if not opt.freeze_visual:
        param_groups.insert(0, {'params': net_visual.parameters(), 'lr': opt.lr_visual})
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    raise ValueError('Unknown --optimizer %r' % opt.optimizer)


def cuda_synchronize():
    # torch.cuda.synchronize() initialises CUDA, which fails on a CPU-only run.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_checkpoint(net_visual, net_unet, optimizer, total_batches, best_err, opt, tag):
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    utils.mkdirs(ckpt_dir)
    torch.save(net_visual.state_dict(), os.path.join(ckpt_dir, 'visual_%s.pth' % tag))
    torch.save(net_unet.state_dict(), os.path.join(ckpt_dir, 'unet_%s.pth' % tag))
    # Written for every tag, not just 'latest': the best model is saved after
    # the latest one, so a sidecar refreshed only on 'latest' would still hold
    # the pre-validation best_err and a resumed run would then overwrite
    # unet_best.pth with a worse model. Holds the optimizer state and the
    # lr_steps / best-error bookkeeping --continue_train needs.
    torch.save({'net_visual': net_visual.state_dict(),
                'net_unet': net_unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'total_batches': total_batches,
                'best_err': best_err},
               os.path.join(ckpt_dir, 'training_state.pth'))

#decreae learning rate
def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#print learning rate
def print_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

def save_visualization(vis_rows, outputs, batch_data, save_dir, opt):
    # fetch data and predictions
    mag_mix = batch_data['audio_mix_mags']
    phase_mix = batch_data['audio_mix_phases']

    pred_masks_ = outputs['pred_mask']
    gt_masks_ = outputs['gt_mask']
    mag_mix_ = outputs['audio_mix_mags']
    weight_ = outputs['weight']

    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, gt_masks_.size(3), warp=False)).to(opt.device)
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
        gt_masks_linear = F.grid_sample(gt_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
        gt_masks_linear = gt_masks_

    # convert into numpy
    mag_mix = mag_mix.detach().cpu().numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.detach().cpu().numpy()
    pred_masks_ = pred_masks_.detach().cpu().numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    gt_masks_ = gt_masks_.detach().cpu().numpy()
    gt_masks_linear = gt_masks_linear.detach().cpu().numpy()
    weight_ = None if weight_ is None else weight_.detach().cpu().numpy()

    # loop over each example
    for j in range(min(B, opt.num_visualization_examples)):
        row_elements = []

        # one folder per example
        prefix = 'example-%d' % j
        utils.mkdirs(os.path.join(save_dir, prefix))

        # save mixture
        mix_wav = utils.istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=opt.stft_hop)
        mix_amp = utils.magnitude2heatmap(mag_mix_[j, 0])
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        imsave(os.path.join(save_dir, filename_mixmag), mix_amp[::-1, :, :])
        wavfile.write(os.path.join(save_dir, filename_mixwav), opt.audio_sampling_rate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # GT and predicted audio reconstruction
        gt_mag = mag_mix[j, 0] * gt_masks_linear[j, 0]
        gt_wav = utils.istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=opt.stft_hop)
        pred_mag = mag_mix[j, 0] * pred_masks_linear[j, 0]
        preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=opt.stft_hop)

        # output masks
        filename_gtmask = os.path.join(prefix, 'gtmask.jpg')
        filename_predmask = os.path.join(prefix, 'predmask.jpg')
        gt_mask = (np.clip(gt_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        pred_mask = (np.clip(pred_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(save_dir, filename_gtmask), gt_mask[::-1, :])
        imsave(os.path.join(save_dir, filename_predmask), pred_mask[::-1, :])

        # ouput spectrogram (log of magnitude, show colormap)
        filename_gtmag = os.path.join(prefix, 'gtamp.jpg')
        filename_predmag = os.path.join(prefix, 'predamp.jpg')
        gt_mag = utils.magnitude2heatmap(gt_mag)
        pred_mag = utils.magnitude2heatmap(pred_mag)
        imsave(os.path.join(save_dir, filename_gtmag), gt_mag[::-1, :, :])
        imsave(os.path.join(save_dir, filename_predmag), pred_mag[::-1, :, :])

        # output audio
        filename_gtwav = os.path.join(prefix, 'gt.wav')
        filename_predwav = os.path.join(prefix, 'pred.wav')
        wavfile.write(os.path.join(save_dir, filename_gtwav), opt.audio_sampling_rate, gt_wav)
        wavfile.write(os.path.join(save_dir, filename_predwav), opt.audio_sampling_rate, preds_wav)

        row_elements += [
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        if weight_ is not None:
            filename_weight = os.path.join(prefix, 'weight.jpg')
            weight = utils.magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
            imsave(os.path.join(save_dir, filename_weight), weight[::-1, :])
            row_elements += [{'image': filename_weight}]

        vis_rows.append(row_elements)

#used to display validation loss
def display_val(model, crit, writer, index, dataset_val_loader, opt):
        # remove previous viz results
        save_dir = os.path.join('.', opt.checkpoints_dir, opt.name, 'visualization')
        utils.mkdirs(save_dir)

        #initial results lists
        coseparation_losses = []

        # initialize HTML header
        visualizer = viz.HTMLVisualizer(os.path.join(save_dir, 'index.html'))
        visualizer.add_header(['Filename', 'Input Mixed Audio', 'Predicted Audio',
                               'GroundTruth Audio', 'Predicted Mask', 'GroundTruth Mask',
                               'Loss weighting'])
        vis_rows = []

        with torch.no_grad():
            for i, val_data in enumerate(dataset_val_loader):
                if i >= opt.validation_batches:
                    # The val loader is built with one batch beyond
                    # validation_batches, so this branch is reachable exactly
                    # when --validation_visualization is set.
                    if opt.validation_visualization:
                        save_visualization(vis_rows, model(val_data), val_data, save_dir, opt)
                    break
                output = model(val_data)
                coseparation_loss = get_coseparation_loss(output, opt, crit['loss_coseparation']) * opt.coseparation_loss_weight
                coseparation_losses.append(coseparation_loss.item())

        avg_coseparation_loss = sum(coseparation_losses)/len(coseparation_losses)
        if vis_rows:
            visualizer.add_rows(vis_rows)
            visualizer.write_html()
        # The co-separation loss is the only validation signal left; the old
        # accuracy and classifier numbers were dropped with the classifier.
        if opt.tensorboard:
            writer.add_scalar('data/val_coseparation_loss', avg_coseparation_loss, index)
        print('val coseparation loss: %.4f' % avg_coseparation_loss)
        return avg_coseparation_loss

def get_coseparation_loss(output, opt, loss_coseparation):
        #initialize a dic to store the index of the list
        vid_index_dic ={}
        vids = output['vids'].squeeze(1).cpu().numpy()
        O = vids.shape[0]
        count = 0
        for i in range(O):
            if vids[i] not in vid_index_dic:
                vid_index_dic[vids[i]] = count
                count = count + 1

        #initialize three lists of length = number of video clips to reconstruct
        predicted_mask_list = [None for i in range(len(vid_index_dic.keys()))]
        gt_mask_list = [None for i in range(len(vid_index_dic.keys()))]
        weight_list = [None for i in range(len(vid_index_dic.keys()))] if opt.weighted_loss else None

        #iterate through all objects
        gt_masks = output['gt_mask']
        mask_prediction = output['pred_mask']
        # None unless --weighted_loss; weight_list stays None in that case so
        # BaseLoss falls back to uniform weights instead of a list of Nones.
        weight = output['weight']

        for i in range(O):
            if predicted_mask_list[vid_index_dic[vids[i]]] is None:
                gt_mask_list[vid_index_dic[vids[i]]] = gt_masks[i,:,:,:]
                if weight_list is not None:
                    weight_list[vid_index_dic[vids[i]]] = weight[i,:,:,:]
                predicted_mask_list[vid_index_dic[vids[i]]] = mask_prediction[i,:,:,:]
            else:
                predicted_mask_list[vid_index_dic[vids[i]]] = predicted_mask_list[vid_index_dic[vids[i]]] + mask_prediction[i,:,:,:]

        if opt.mask_loss_type == 'BCE':
            for i in range(O):
                #clip the prediction results to make it in the range of [0,1] for BCE loss
                predicted_mask_list[vid_index_dic[vids[i]]] = torch.clamp(predicted_mask_list[vid_index_dic[vids[i]]], 0, 1)
        coseparation_loss = loss_coseparation(predicted_mask_list, gt_mask_list, weight_list)
        return coseparation_loss


#parse arguments (extra flags are registered before parsing)
options = TrainOptions()
options.initialize()
options.parser.add_argument('--freeze_visual', action='store_true',
                            help='freeze the visual stream and optimise only the UNet')
options.parser.add_argument('--auto_split', action='store_true',
                            help='regenerate train.txt/val.txt from --data_path before training')
options.parser.add_argument('--split_dir', type=str, default='',
                            help='where the generated split lists go (default: checkpoints_dir/name/splits)')
options.parser.add_argument('--val_ratio', type=float, default=0.1,
                            help='fraction of videos held out when --auto_split')
opt = options.parse()
opt.device = torch.device('cuda:%d' % opt.gpu_ids[0]) if opt.gpu_ids else torch.device('cpu')

#build split lists if needed, then point the dataset at them
split_dir = opt.split_dir or os.path.join(opt.checkpoints_dir, opt.name, 'splits')
if opt.auto_split:
        counts, n_videos, n_val = build_split_lists(opt.data_path, split_dir, opt.val_ratio, opt.seed)
        print('generated splits in %s: %d videos (%d held out) | %s'
              % (split_dir, n_videos, n_val, counts))
opt.hdf5_path = split_dir

# The dataset reads <mode>.txt from hdf5_path when a split file is prebuilt,
# so fail early with a useful message instead of a bare FileNotFoundError deep
# inside initialize(). --auto_split always writes both files, so this only
# guards the case where an existing split dir is reused.
if not opt.auto_split:
    needed = [opt.mode + '.txt'] + (['val.txt'] if opt.validation_on else [])
    for name in needed:
        path = os.path.join(opt.hdf5_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                '%s not found. Either pass --auto_split to build the split lists '
                'from --data_path, or point --split_dir at a directory holding %s.'
                % (path, ' and '.join(needed)))

#construct data loader
dataset, dataset_loader = create_loader(opt, num_workers=int(opt.nThreads))
print('#training samples = %d' % len(dataset))

#create validation set data loader if validation_on option is set.
# A shallow copy keeps the val dataset from writing its mode back into the
# training namespace, which would switch augmentation off inside the workers.
if opt.validation_on:
        opt_val = copy.copy(opt)
        opt_val.mode = 'val'
        if opt.validation_visualization:
                # one extra batch beyond validation_batches feeds save_visualization
                opt_val.validation_batches = opt.validation_batches + 1
        dataset_val, dataset_val_loader = create_loader(opt_val, num_workers=2)
        print('#validation samples = %d' % len(dataset_val))

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# Network Builders: only the two branches survive
builder = ModelBuilder()
net_visual = builder.build_visual(
        pool_type=opt.visual_pool,
        fc_out = 256,
        weights=opt.weights_visual)
net_unet = builder.build_unet(
        unet_num_layers = opt.unet_num_layers,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_unet)
if opt.freeze_visual:
        utils.set_requires_grad([net_visual], False)
# The two branches, in the order AudioVisualModel unpacks them.
nets = (net_visual, net_unet)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model.to(opt.device)

# Set up optimizer
optimizer = create_optimizer(nets, opt)

# Set up loss functions
if opt.mask_loss_type == 'L1':
    loss_coseparation = criterion.L1Loss()
elif opt.mask_loss_type == 'L2':
    loss_coseparation = criterion.L2Loss()
elif opt.mask_loss_type == 'BCE':
    loss_coseparation = criterion.BCELoss()
else:
    raise ValueError('Unknown --mask_loss_type %r' % opt.mask_loss_type)
if(len(opt.gpu_ids) > 0):
    loss_coseparation.cuda(opt.gpu_ids[0])

crit = {'loss_coseparation': loss_coseparation}
#initialization
total_batches = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_coseparation_loss = []
best_err = float("inf")

# Resume before the first batch so the restored counters and lr schedule are in
# force for the whole run.
if opt.continue_train:
    state_path = os.path.join('.', opt.checkpoints_dir, opt.name, 'training_state.pth')
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            '--continue_train needs %s; run at least one --save_latest_freq '
            'interval first.' % state_path)
    state = torch.load(state_path, map_location=opt.device)
    net_visual.load_state_dict(state['net_visual'])
    net_unet.load_state_dict(state['net_unet'])
    optimizer.load_state_dict(state['optimizer'])
    total_batches = state['total_batches']
    best_err = state['best_err']
    print('resumed from %s at total_batches %d (best_err %.5f)'
          % (state_path, total_batches, best_err))

for epoch in range(1 + opt.epoch_count, opt.niter+1):
        cuda_synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()
        for i, data in enumerate(dataset_loader):
                # `data` is the collated dict from object_collate; iterating the
                # raw Dataset would yield per-sample numpy instead of tensors.
                if(opt.measure_time):
                    cuda_synchronize()
                    iter_data_loaded_time = time.time()

                #forward pass
                model.zero_grad()
                output = model(data)

                # The co-separation mask loss is the only objective left
                coseparation_loss = get_coseparation_loss(output, opt, loss_coseparation) * opt.coseparation_loss_weight

                if(opt.measure_time):
                    cuda_synchronize()
                    iter_data_forwarded_time = time.time()
                batch_coseparation_loss.append(coseparation_loss.item())

                optimizer.zero_grad()
                coseparation_loss.backward()
                optimizer.step()

                if(opt.measure_time):
                        cuda_synchronize()
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                total_batches += 1

                if(total_batches % opt.display_freq == 0):
                        print('Display training progress at (epoch %d, total_batches %d)' % (epoch, total_batches))
                        avg_coseparation_loss = sum(batch_coseparation_loss)/len(batch_coseparation_loss)
                        print('co-separation loss: %.5f' % avg_coseparation_loss)
                        batch_coseparation_loss = []
                        if opt.tensorboard:
                            writer.add_scalar('data/coseparation_loss', avg_coseparation_loss, total_batches)

                        if(opt.measure_time):
                                print('average data loading time: %.3f' % (sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: %.3f' % (sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: %.3f' % (sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        print('end of display \n')

                if(total_batches % opt.save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_batches %d)' % (epoch, total_batches))
                        save_checkpoint(net_visual, net_unet, optimizer, total_batches, best_err, opt, 'latest')
                        print('Latest learning rate:')
                        print_learning_rate(optimizer)
                if(total_batches % opt.validation_freq == 0 and opt.validation_on):
                        model.eval()
                        print('Display validation results at (epoch %d, total_batches %d)' % (epoch, total_batches))
                        val_err = display_val(model, crit, writer, total_batches, dataset_val_loader, opt)
                        print('end of display \n')
                        model.train()
                        #save the model that achieves the smallest validation error
                        if val_err < best_err:
                            best_err = val_err
                            print('saving the best model (epoch %d, total_batches %d) with validation error %.5f\n' % (epoch, total_batches, val_err))
                            save_checkpoint(net_visual, net_unet, optimizer, total_batches, best_err, opt, 'best')
                #decrease learning rate
                if(total_batches in opt.lr_steps):
                        decrease_learning_rate(optimizer, opt.decay_factor)
                        print('decreased learning rate by ', opt.decay_factor)

                if(opt.measure_time):
                        cuda_synchronize()
                        iter_start_time = time.time()

print('training done: %d batches' % total_batches)
