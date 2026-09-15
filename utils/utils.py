from distutils.command.clean import clean
import os
import shutil
import librosa
import torch
import cv2
import numpy as np
string_classes = (str, bytes)
import collections.abc as container_abcs
import matplotlib.pyplot as plt
import array
import random
import wave
plt.switch_backend('Agg') 
plt.ioff()

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color

def mkdirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path)

def visualizeSpectrogram(spectrogram, save_path):
	fig,ax = plt.subplots(1,1)
	plt.axis('off')
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	plt.pcolormesh(librosa.amplitude_to_db(spectrogram))
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def istft_reconstruction(mag, phase, hop_length=256, length=65535):
    spec = mag.astype(complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length, length=length)
    return np.clip(wav, -1., 1.)


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
	nets (network list)   -- a list of networks
	requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


#define customized collate to combine useful objects across video pairs
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
def object_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    #print batch
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            return torch.cat([torch.from_numpy(b) for b in batch], 0) #concatenate even if dimension differs
            #return object_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: object_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [object_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()


from random import randrange
def sample_audio(audio, window):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    return audio_sample

def mix(clean_file,noise_file,snr):

    clean_amp = clean_file
    noise_amp = noise_file

    clean_rms = cal_rms(clean_amp)
    
    start = random.randint(0, abs(len(noise_amp)-len(clean_amp)))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(divided_noise_amp)
    
    
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms) 

    mixed_amp = (clean_amp + adjusted_noise_amp)
    
    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max 
    if  mixed_amp.max(axis=0) > max_int16:
        reduction_rate = max_int16 / mixed_amp.max(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
    return mixed_amp

def mix_2(clean_file,noise_file,output_mixed_file,snr):
    clean_amp = clean_file
    noise_amp = noise_file

    clean_rms = cal_rms(clean_amp)
    
    start = random.randint(0, abs(len(noise_amp)-len(clean_amp)))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(divided_noise_amp)
    
    
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms) 

    mixed_amp = (clean_amp + adjusted_noise_amp)
    
    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max 
    if  mixed_amp.max(axis=0) > max_int16:
        reduction_rate = max_int16 / mixed_amp.max(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
    save_waveform(output_mixed_file, clean_wav.getparams(), mixed_amp)