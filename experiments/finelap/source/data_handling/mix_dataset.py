import math
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import torchaudio
from torchaudio.transforms import Resample
from utils.logger_config import main_logger
from models.feature_extractor import EAT_preprocess
from data_handling.data_utils import pad_sequence
from torchlibrosa.augmentation import SpecAugmentation

PHRASE_BANK = './data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl'


class MixDataset(Dataset):
    """
    Mixed dataset for both retrieval and grounding 
    Data format: 
    - With grounding: {"audio_id": audio_id, "audio_path": audio_path, "caption": caption, "phrases": [{"phrase": phrase, "segments": [onset, offset]}]}
    - Without grounding: {"audio_id": audio_id, "audio_path": audio_path, "caption": caption}
    """

    def __init__(self,
                 metadata_files,
                 sample_rate: int = 32000,
                 max_length: int = 10,
                 time_resolution: float = 10/32,
                 use_alternatives: bool = True,
                 max_phrases: int = 20,
                 return_type: str = "raw",  # raw or mel
                 spec_aug: bool = False, 
                 max_split_segments: int = 1, # split wav into N segments, for distillation loss 
                 max_chosen_segments: int = 1, # max number of segments to choose from each wav
                 **kwargs):  
        
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.time_resolution = time_resolution
        self.use_alternatives = use_alternatives
        self.max_phrases = max_phrases
        self.dataset_name = kwargs.get("dataset_name", "")
        self.return_type = return_type
        self.spec_aug = spec_aug
        self.N_split = max_split_segments
        self.N_chosen = max_chosen_segments
        self.waveform_aug = bool(kwargs.get("waveform_aug", False))
        self.waveform_aug_prob = float(kwargs.get("waveform_aug_prob", 0.0))
        self.noise_aug_prob = float(kwargs.get("noise_aug_prob", 0.0))
        self.gaussian_noise_prob = float(kwargs.get("gaussian_noise_prob", 0.0))
        self.noise_snr_min = float(kwargs.get("noise_snr_min", 5.0))
        self.noise_snr_max = float(kwargs.get("noise_snr_max", 20.0))
        self.noise_path = kwargs.get("noise_path", "")
        self.reverb_aug_prob = float(kwargs.get("reverb_aug_prob", 0.0))
        self.rir_path = kwargs.get("rir_path", "")
        self.noise_files = self._collect_audio_files(self.noise_path)
        self.rir_files = self._collect_audio_files(self.rir_path)
        self.phrase_bank_path = kwargs.get("phrase_bank", PHRASE_BANK)
        self.random_crop = bool(kwargs.get("random_crop", False))
        self.strict_audio = bool(kwargs.get("strict_audio", False))
        self.audio_backend = kwargs.get("audio_backend", None)
        self.warn_unknown_phrases = bool(kwargs.get("warn_unknown_phrases", True))
        assert self.N_chosen <= self.N_split, f"N_chosen {self.N_chosen} must be less than or equal to N_split {self.N_split}"
        if self.N_chosen != 1 and self.N_split != 1:
            assert self.return_type == "mel", f"return_type {self.return_type} must be mel when N_chosen {self.N_chosen} and N_split {self.N_split} are not 1"

        with open(self.phrase_bank_path, 'r') as f:
            self.phrase_bank = [json.loads(line) for line in f]
            
        main_logger.info(f"Max phrases for sample: {max_phrases}")
        
        # cluster is the main phrase name, alternatives include the cluster itself and variants
        self.cluster2alternatives = {phrase['phrase']: phrase['alternatives'] for phrase in self.phrase_bank}
        self.alternative2cluster = {alt: phrase['phrase'] for phrase in self.phrase_bank for alt in phrase['alternatives']}
        self.all_clusters = list(self.cluster2alternatives.keys())
        self.all_phrases = list(self.alternative2cluster.keys())

        data_with_grounding = []
        data_without_grounding = []
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    phrases = item.get("phrases", [])
                    has_valid_grounding = False
                    if phrases:
                        for phrase_item in phrases:
                            if phrase_item.get("segments") and len(phrase_item["segments"]) > 0:
                                has_valid_grounding = True
                                break
                    
                    if has_valid_grounding:
                        data_with_grounding.append(item)
                    else:
                        data_without_grounding.append(item)
        
        self.data = data_with_grounding + data_without_grounding
        self.num_with_grounding = len(data_with_grounding)
        main_logger.info(f'Number of clusters: {len(self.all_clusters)}, number of all phrases: {len(self.all_phrases)}')
        main_logger.info(f"Number of data with grounding: {self.num_with_grounding}, number of data without grounding: {len(data_without_grounding)}, total number of data: {len(self.data)}")
        if self.N_chosen != 1 and self.N_split != 1: 
            main_logger.info(f"Max split segments: {self.N_split}, max chosen segments: {self.N_chosen}")
        if self.waveform_aug:
            main_logger.info(
                "Waveform augmentation enabled: "
                f"prob={self.waveform_aug_prob}, noise_prob={self.noise_aug_prob}, "
                f"gaussian_prob={self.gaussian_noise_prob}, reverb_prob={self.reverb_aug_prob}, "
                f"noise_files={len(self.noise_files)}, rir_files={len(self.rir_files)}"
            )

    @staticmethod
    def _collect_audio_files(root):
        if not root:
            return []
        if not os.path.exists(root):
            return []
        audio_files = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                    audio_files.append(os.path.join(dirpath, filename))
        return audio_files

    def _load_resampled_mono(self, path):
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        return waveform

    @staticmethod
    def _match_length(waveform, length):
        if waveform.numel() >= length:
            max_start = waveform.numel() - length
            start = random.randint(0, max_start) if max_start > 0 else 0
            return waveform[start:start + length]
        repeats = int(math.ceil(length / max(1, waveform.numel())))
        return waveform.repeat(repeats)[:length]

    def _add_noise(self, waveform):
        if self.noise_files and random.random() < self.noise_aug_prob:
            try:
                noise = self._load_resampled_mono(random.choice(self.noise_files))
                noise = self._match_length(noise, waveform.numel()).to(waveform.dtype)
            except Exception:
                noise = torch.randn_like(waveform)
        elif random.random() < self.gaussian_noise_prob:
            noise = torch.randn_like(waveform)
        else:
            return waveform

        signal_power = waveform.pow(2).mean().clamp_min(1e-10)
        noise_power = noise.pow(2).mean().clamp_min(1e-10)
        snr = random.uniform(self.noise_snr_min, self.noise_snr_max)
        scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr / 10.0))))
        return waveform + noise * scale

    def _apply_reverb(self, waveform):
        if random.random() >= self.reverb_aug_prob:
            return waveform
        if self.rir_files:
            try:
                rir = self._load_resampled_mono(random.choice(self.rir_files)).to(waveform.dtype)
            except Exception:
                return waveform
        else:
            rir_len = random.randint(int(0.05 * self.sample_rate), int(0.25 * self.sample_rate))
            decay = torch.exp(-torch.linspace(0, random.uniform(4.0, 8.0), rir_len))
            rir = torch.randn(rir_len, dtype=waveform.dtype) * decay
            rir[0] = 1.0
        rir = rir / rir.abs().max().clamp_min(1e-8)
        max_rir = min(rir.numel(), int(0.5 * self.sample_rate))
        rir = rir[:max_rir]
        wet = F.conv1d(waveform.view(1, 1, -1), rir.flip(0).view(1, 1, -1), padding=rir.numel() - 1)
        wet = wet.view(-1)[:waveform.numel()]
        wet = wet / wet.abs().max().clamp_min(1e-8) * waveform.abs().max().clamp_min(1e-8)
        return 0.7 * waveform + 0.3 * wet

    def _augment_waveform(self, waveform):
        if not self.waveform_aug or random.random() >= self.waveform_aug_prob:
            return waveform
        augmented = self._add_noise(waveform)
        augmented = self._apply_reverb(augmented)
        return augmented.clamp(-1.0, 1.0)

    def _sample_alternative(self, phrase):
        """Sample a random alternative for a phrase (cluster or alternative)"""
        cluster = self.alternative2cluster.get(phrase, phrase)
        alternatives = self.cluster2alternatives.get(cluster, [phrase])
        return random.choice(alternatives)
    
    def _sample_negative_phrases(self, positive_clusters):
        """Sample negative phrase clusters and their alternatives"""
        excluded_clusters = set(positive_clusters)
        available_clusters = [c for c in self.all_clusters if c not in excluded_clusters]
        
        n_positive = len(positive_clusters)
        n_negative = max(0, self.max_phrases - n_positive)
        
        if n_negative == 0 or not available_clusters:
            return []
        
        n_negative = min(n_negative, len(available_clusters))
        negative_clusters = random.sample(available_clusters, n_negative)
        
        if self.use_alternatives:
            negative_phrases = [self._sample_alternative(cluster) for cluster in negative_clusters]
        else:
            negative_phrases = negative_clusters
        
        return negative_phrases

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        caption = audio_item.get("caption", "")
        if type(caption) != str and type(caption) != list:
            print(f"Invalid caption type: {type(caption)}, caption: {caption}")
            caption = ""
        if isinstance(caption, list):
            caption = random.choice(caption)  # randomly choose one caption from the list
        phrases_data = audio_item.get("phrases", [])
        has_strong_annotation = any(p.get("segments") for p in phrases_data)
        audio_path = audio_item.get("audio_path", "")

        # Load audio
        try: 
            wav_info = torchaudio.info(audio_path, backend=self.audio_backend)
            crop_frames = int(self.max_length * wav_info.sample_rate)
            max_start = max(0, wav_info.num_frames - crop_frames)
            start_frame = random.randint(0, max_start) if self.random_crop else 0
            waveform, sr = torchaudio.load(
                audio_path, frame_offset=start_frame, num_frames=crop_frames, backend=self.audio_backend
            )
            crop_start = start_frame / wav_info.sample_rate
            phrases_data = [
                dict(p, segments=[
                    [max(0.0, float(a) - crop_start), min(self.max_length, float(b) - crop_start)]
                    for a, b in p.get("segments", [])
                    if float(b) > crop_start and float(a) < crop_start + self.max_length
                    and float(b) > float(a)
                ]) for p in phrases_data
            ]
        except (FileNotFoundError, RuntimeError) as e: 
            if self.strict_audio:
                raise RuntimeError(f"Unable to load training audio: {audio_path}") from e
            waveform = torch.zeros((1, 32000))
            sr = 32000  
            print(f"{audio_path} not found")

        if waveform.shape[-1] < 0.1*self.sample_rate:
            waveform = torch.zeros((1, self.max_length*self.sample_rate))
            print(f"Audio too short: {audio_path}")
        else: 
            waveform = waveform[0]
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)  
            waveform = self._augment_waveform(waveform)

        processed_phrases = []
        if phrases_data:
            positive_clusters = []
            for phrase_item in phrases_data[:self.max_phrases]:
                phrase = phrase_item.get("phrase", "")
                if not phrase:
                    continue
                if self.warn_unknown_phrases and phrase not in self.alternative2cluster:
                    # main_logger.warning(f"Phrase {phrase} not found in phrase bank")
                    print(f"Phrase {phrase} not found in phrase bank")
                cluster = self.alternative2cluster.get(phrase, phrase)
                positive_clusters.append(cluster)
                
                if self.use_alternatives:
                    sampled_phrase = self._sample_alternative(phrase)
                else:
                    sampled_phrase = phrase
                
                processed_phrases.append({
                    "phrase": sampled_phrase,
                    "segments": phrase_item.get("segments", []),
                    "is_positive": True
                })
            
            negative_phrases = self._sample_negative_phrases(positive_clusters)
            for neg_phrase in negative_phrases:
                processed_phrases.append({
                    "phrase": neg_phrase,
                    "segments": [],
                    "is_positive": False
                })
        if self.return_type == "mel":
            audio = EAT_preprocess(waveform=waveform)  # already padded to 1024 frames
            audio_chosen_segments = None
            dense_audio_embeds_idx = None

            if self.N_split != 1:
                T, F = audio.shape[-2], audio.shape[-1]
                T_seg = T // self.N_split

                split_segments = audio.reshape(1, self.N_split, T_seg, F).squeeze(0)  # [N_split, T_seg, F]
                split_segments = split_segments.unsqueeze(1)  # [N_split, 1, T_seg, F]

                chosen_indices = random.sample(range(self.N_split), self.N_chosen)
                audio_chosen_segments = split_segments[chosen_indices]  # [N_chosen, 1, T_seg, F]

                num_embeds = 64  # ad-hoc n_embeds for EAT
                embeds_per_segment = num_embeds // self.N_split
                dense_audio_embeds_idx = []
                for idx in chosen_indices:
                    start_embed = idx * embeds_per_segment
                    end_embed = (idx + 1) * embeds_per_segment if idx < self.N_split - 1 else num_embeds
                    segment_indices = list(range(start_embed, end_embed))
                    dense_audio_embeds_idx.append(segment_indices)
                dense_audio_embeds_idx = torch.tensor(dense_audio_embeds_idx)  # [N_chosen, embeds_per_segment]

            if self.spec_aug:
                spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                  time_stripes_num=2,
                                                  freq_drop_width=8,
                                                  freq_stripes_num=2)
                audio = spec_augmenter(audio)
        elif self.return_type == "raw":
            audio = waveform
            audio_chosen_segments = None
            dense_audio_embeds_idx = None # distillation loss not implemented for raw waveform
        else: 
            raise ValueError(f"Invalid return type: {self.return_type}")

        return {
            "audio_id": audio_id,
            "audio": audio, # waveform or mel
            "caption": caption,
            "phrases": processed_phrases,
            "has_strong_annotation": has_strong_annotation if self.random_crop else False,
            "audio_chosen_segments": audio_chosen_segments,  # [N_chosen, 1, T_seg, F]
            "dense_audio_embeds_idx": dense_audio_embeds_idx,
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data_batch):
        audio_ids = []
        audios = []
        captions = []
        batch_phrases = []
        batch_frame_labels = []
        has_grounding = []
        batch_audio_chosen_segments = []
        batch_dense_audio_embeds_idx = []

        n_frames = int(self.max_length / self.time_resolution)

        for data in data_batch:
            audio_ids.append(data["audio_id"])
            audios.append(data["audio"])
            captions.append(data["caption"])

            phrases = data.get("phrases", [])
            has_valid_grounding = data.get("has_strong_annotation", False)
            phrase_names = []
            frame_labels = []

            if phrases:
                for phrase_item in phrases:
                    phrase = phrase_item.get("phrase", "")
                    segments = phrase_item.get("segments", [])
                    is_positive = phrase_item.get("is_positive", True)

                    if not phrase:
                        continue

                    phrase_names.append(phrase)

                    label = np.zeros(n_frames, dtype=int)
                    if is_positive and segments:
                        has_valid_grounding = True
                        for start, end in segments:
                            onset = int(start / self.time_resolution)
                            offset = int(end / self.time_resolution)
                            onset = max(0, min(onset, n_frames-1))
                            offset = max(0, min(offset, n_frames))
                            if onset < offset:
                                label[onset:offset] = 1
                    frame_labels.append(label)

            if has_valid_grounding and phrase_names:
                frame_labels_array = np.array(frame_labels)
                frame_labels_tensor = torch.tensor(frame_labels_array, dtype=torch.int)
            else:
                phrase_names = None
                frame_labels_tensor = None

            batch_phrases.append(phrase_names)
            batch_frame_labels.append(frame_labels_tensor)
            has_grounding.append(has_valid_grounding)

            audio_chosen_segments = data.get("audio_chosen_segments")
            dense_audio_embeds_idx = data.get("dense_audio_embeds_idx")
            batch_audio_chosen_segments.append(audio_chosen_segments)
            batch_dense_audio_embeds_idx.append(dense_audio_embeds_idx)

        # Pad waveforms or Stack mel
        if self.return_type == "raw":
            padded_audios, waveform_lengths = pad_sequence(audios)
        elif self.return_type == "mel":
            padded_audios = torch.cat(audios, dim=0)
        else:
            raise ValueError(f"Invalid return type: {self.return_type}")

        if self.N_split != 1:
            batch_audio_chosen_segments = torch.stack(batch_audio_chosen_segments, dim=0)  # [bsz, N_chosen, 1, T_seg, F]
            batch_dense_audio_embeds_idx = torch.stack(batch_dense_audio_embeds_idx, dim=0)
        else:
            batch_audio_chosen_segments = None
            batch_dense_audio_embeds_idx = None

        return {
            "audio_id": audio_ids,
            "audios": padded_audios,
            "caption": captions,
            "phrases": batch_phrases,
            "frame_labels": batch_frame_labels,
            "has_grounding": has_grounding,
            "batch_audio_chosen_segments": batch_audio_chosen_segments,  # [bsz, N_chosen, 1, T_seg, F]
            "dense_audio_embeds_idx": batch_dense_audio_embeds_idx,  # [bsz, N_chosen, embeds_per_segment]
        }
