import codecs as cs
import os
import random
from os.path import dirname, join as pjoin

import numpy as np
from rich.progress import track
from torch.utils import data


def _canonicalize_imfs(imfs: np.ndarray) -> np.ndarray:
    if imfs.ndim == 3 and imfs.shape[0] == 3 and imfs.shape[1] == 63:
        return imfs
    if imfs.ndim == 3 and imfs.shape[0] == 3 and imfs.shape[1] == 69:
        return imfs
    if imfs.ndim == 3 and imfs.shape[0] == 3 and imfs.shape[2] == 63:
        return imfs.transpose(0, 2, 1)
    if imfs.ndim == 3 and imfs.shape[0] == 3 and imfs.shape[2] == 69:
        return imfs.transpose(0, 2, 1)
    if imfs.ndim == 4 and imfs.shape[0] == 3 and imfs.shape[3] == 3:
        t = imfs.shape[1]
        dof = imfs.shape[2] * imfs.shape[3]
        return imfs.reshape(3, t, dof).transpose(0, 2, 1)
    if imfs.ndim == 4 and imfs.shape[0] == 3 and imfs.shape[2] == 3:
        t = imfs.shape[3]
        dof = imfs.shape[1] * imfs.shape[2]
        return imfs.reshape(3, dof, t)
    raise ValueError(f"Unsupported IMF shape: {tuple(imfs.shape)}")


class Text2MotionDatasetV2HHT(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.debug = debug
        self.mean = mean
        self.std = std
        self.motion_dir = motion_dir
        self.text_dir = text_dir

        cfg = kwargs.get("cfg", None)
        self.data_cfg = getattr(cfg, "DATA", None)
        self.use_offline_imf = bool(getattr(self.data_cfg, "USE_OFFLINE_IMF_GT", False))

        data_root = dirname(motion_dir)
        self.imf_root = self._resolve_imf_root(data_root)
        self.imf_dir = self._resolve_imf_dir(self.imf_root)
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    id_list.append(line)
        self.id_list = id_list
        self.mean_imf = None
        self.std_imf = None
        if self.use_offline_imf:
            self.mean_imf, self.std_imf = self._load_or_compute_imf_stats()

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        enumerator = enumerate(
            track(id_list, f"Loading HHT {split_file.split('/')[-1].split('.')[0]}")
        ) if progress_bar else enumerate(id_list)

        data_dict = {}
        new_name_list = []
        length_list = []
        self.nfeats = None
        count = 0

        for _, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if self.nfeats is None:
                    self.nfeats = motion.shape[1]
                if len(motion) < self.min_motion_length or len(motion) >= 200:
                    continue

                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        if len(line_split) < 2:
                            continue
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2]) if len(line_split) > 2 else 0.0
                        to_tag = float(line_split[3]) if len(line_split) > 3 else 0.0
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict = {"caption": caption, "tokens": tokens}
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                start = int(f_tag * 20)
                                end = int(to_tag * 20)
                                n_motion = motion[start:end]
                                if len(n_motion) < self.min_motion_length or len(n_motion) >= 200:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "source_id": name,
                                    "segment_range": (start, end),
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except Exception:
                                pass

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "source_id": name,
                        "segment_range": None,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except Exception:
                continue

        if self.nfeats is None:
            raise ValueError("No valid motion files were loaded for HHT dataset.")

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def _resolve_imf_root(self, data_root):
        if not self.use_offline_imf:
            return data_root
        explicit_root = getattr(self.data_cfg, "IMF_ROOT", None)
        if explicit_root:
            return explicit_root
        explicit_dir = getattr(self.data_cfg, "IMF_DIR", None)
        if explicit_dir:
            return dirname(explicit_dir.rstrip("/"))
        sibling_name = getattr(self.data_cfg, "IMF_ROOT_NAME", None)
        if sibling_name:
            return pjoin(dirname(data_root), sibling_name)
        return data_root

    def _resolve_imf_dir(self, imf_root):
        explicit_dir = getattr(self.data_cfg, "IMF_DIR", None)
        if explicit_dir:
            return explicit_dir
        candidates = [
            getattr(self.data_cfg, "IMF_DIR_NAME", "joints_imf"),
            "joints_imf",
            "aligned3/imfs3",
        ]
        for rel in candidates:
            cand = pjoin(imf_root, rel)
            if os.path.isdir(cand):
                return cand
        return pjoin(imf_root, candidates[0])

    def _candidate_imf_paths(self, sample_id):
        candidates = [pjoin(self.imf_dir, sample_id + ".npy")]
        if "__s" not in sample_id:
            candidates.extend(
                [
                    pjoin(self.imf_dir, sample_id + "__s000000_e000197.npy"),
                    pjoin(self.imf_dir, sample_id + "__s000000_e000196.npy"),
                ]
            )
        return candidates

    def _load_or_compute_imf_stats(self):
        mean_name = getattr(self.data_cfg, "IMF_MEAN_NAME", "Mean_imf.npy")
        std_name = getattr(self.data_cfg, "IMF_STD_NAME", "Std_imf.npy")
        stats_root = getattr(self.data_cfg, "IMF_STATS_ROOT", None) or self.imf_root
        mean_path = pjoin(stats_root, mean_name)
        std_path = pjoin(stats_root, std_name)
        if os.path.exists(mean_path) and os.path.exists(std_path):
            return np.load(mean_path), np.load(std_path)

        sum_imf = None
        sumsq_imf = None
        count = 0
        for sid in self.id_list:
            imf_path = None
            for cand in self._candidate_imf_paths(sid):
                if os.path.exists(cand):
                    imf_path = cand
                    break
            if imf_path is None:
                continue
            imfs = _canonicalize_imfs(np.load(imf_path)).astype(np.float64)
            if imfs.shape[2] > 0:
                cur_sum = imfs.sum(axis=2)
                cur_sumsq = np.square(imfs).sum(axis=2)
                if sum_imf is None:
                    sum_imf = cur_sum
                    sumsq_imf = cur_sumsq
                else:
                    sum_imf += cur_sum
                    sumsq_imf += cur_sumsq
                count += imfs.shape[2]

        if sum_imf is None or count == 0:
            raise FileNotFoundError(
                f"Unable to locate IMF stats or IMF labels under {self.imf_dir}"
            )

        mean_imf = (sum_imf / count).astype(np.float32)
        var_imf = np.maximum(sumsq_imf / count - np.square(mean_imf.astype(np.float64)), 1e-8)
        std_imf = np.sqrt(var_imf).astype(np.float32)
        os.makedirs(stats_root, exist_ok=True)
        np.save(mean_path, mean_imf)
        np.save(std_path, std_imf)
        return mean_imf, std_imf

    def __len__(self):
        return len(self.name_list) - self.pointer

    def _load_imfs(self, source_id, segment_range):
        if not self.use_offline_imf:
            return None
        imf_path = None
        for cand in self._candidate_imf_paths(source_id):
            if os.path.exists(cand):
                imf_path = cand
                break
        if imf_path is None:
            return None
        imfs = _canonicalize_imfs(np.load(imf_path))
        if segment_range is not None:
            start, end = segment_range
            imfs = imfs[:, :, start:end]
        return imfs

    def __getitem__(self, item):
        idx = self.pointer + item
        sample = self.data_dict[self.name_list[idx]]
        motion = sample["motion"]
        m_length = sample["length"]
        text_data = random.choice(sample["text"])
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        coin2 = np.random.choice(["single", "single", "double"]) if self.unit_length < 10 else "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        if m_length <= 0:
            return None

        start = random.randint(0, len(motion) - m_length)
        end = start + m_length
        motion = motion[start:end]
        motion = (motion - self.mean) / self.std
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        batch = {
            "word_embs": word_embeddings.astype(np.float32),
            "pos_ohot": pos_one_hots.astype(np.float32),
            "text": caption,
            "text_len": sent_len,
            "motion": motion.astype(np.float32),
            "length": int(m_length),
            "tokens": "_".join(tokens),
        }

        if self.use_offline_imf:
            imfs = self._load_imfs(sample["source_id"], sample["segment_range"])
            if imfs is None:
                return None
            if imfs.shape[2] == sample["length"] + 1:
                imfs = imfs[:, :, :-1]
            if imfs.shape[2] < end:
                return None
            imfs = imfs[:, :, start:end]
            mean_imf = self.mean_imf[:, :, None]
            std_imf = np.maximum(self.std_imf[:, :, None], 1e-8)
            batch["imfs"] = ((imfs - mean_imf) / std_imf).astype(np.float32)

        return batch
