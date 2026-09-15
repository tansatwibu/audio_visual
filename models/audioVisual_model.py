import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import warpgrid


class AudioVisualModel(torch.nn.Module):
    """Two-branch co-separation model.

    A visual stream embeds each detected object, and an audio UNet predicts a
    soft mask over the mixture spectrogram conditioned on that embedding. The
    classifier, vocal-attribute and facial streams of the original model are
    gone, so the only training signal is the co-separation loss computed on the
    clip-level sum of the per-object masks.
    """

    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        # The two branches, in the order train.py passes them.
        self.net_visual, self.net_unet = nets[0], nets[1]

    def forward(self, input):
        vids = input['vids']
        audio_mags = input['audio_mags']
        audio_mix_mags = input['audio_mix_mags']
        visuals = input['visuals']
        audio_mix_mags = audio_mix_mags + 1e-10

        # warp both spectrograms onto the same log-frequency grid
        B = audio_mix_mags.size(0)
        T = audio_mix_mags.size(3)
        if self.opt.log_freq:
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
            audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
            audio_mags = F.grid_sample(audio_mags, grid_warp)

        # ground-truth masks, clamped so one loud bin cannot dominate the loss
        gt_masks = audio_mags / audio_mix_mags
        gt_masks.clamp_(0., 5.)

        # visual stream embeds the detected objects
        visual_feature = self.net_visual(Variable(visuals, requires_grad=False))

        # fusion through the UNet predicts the mask. The input spectrogram is
        # detached so gradients only reach the mask branch and never push the
        # network toward re-encoding the mixture itself.
        audio_log_mags = torch.log(audio_mix_mags).detach()
        mask_prediction = self.net_unet(audio_log_mags, visual_feature)

        separated_spectrogram = audio_mix_mags * mask_prediction

        # per-bin loss weighting: louder bins matter more, bounded so silence
        # cannot blow the loss up
        if self.opt.weighted_loss:
            weight = torch.log1p(audio_mix_mags)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = None

        output = {'pred_mask': mask_prediction, 'gt_mask': gt_masks,
                  'pred_spectrogram': separated_spectrogram, 'visual_object': visuals,
                  'audio_mix_mags': audio_mix_mags, 'weight': weight, 'vids': vids}
        return output