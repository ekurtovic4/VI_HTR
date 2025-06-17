"""Dataset reader and process"""

import os
import html
import h5py
import random
import numpy as np
import multiprocessing
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from data import preproc as pp
from functools import partial


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = self._init_dataset()

        for y in self.partitions:
            self.dataset[y]['path'] += dataset[y]['path']
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def save_partitions(self, target, image_input_size, max_text_length):
        """Save images and sentences from dataset"""

        os.makedirs(os.path.dirname(target), exist_ok=True)
        total = 0

        with h5py.File(target, "w") as hf:
            for pt in self.partitions:
                size = (len(self.dataset[pt]['dt']),) + image_input_size[:2]
                total += size[0]

                dummy_image = np.zeros(size, dtype=np.uint8)
                dummy_sentence = [("c" * max_text_length).encode()] * size[0]

                hf.create_dataset(f"{pt}/dt", data=dummy_image, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{pt}/gt", data=dummy_sentence, compression="gzip", compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 1024

        for pt in self.partitions:
            for batch in range(0, len(self.dataset[pt]['gt']), batch_size):
                images = []

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    r = pool.map(partial(pp.preprocess, input_size=image_input_size),
                                 self.dataset[pt]['dt'][batch:batch + batch_size])
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{pt}/dt"][batch:batch + batch_size] = images
                    hf[f"{pt}/gt"][batch:batch + batch_size] = [s.encode() for s in self.dataset[pt]
                                                                ['gt'][batch:batch + batch_size]]
                    pbar.update(batch_size)

    def _init_dataset(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": [], "path": []}

        return dataset

    def _shuffle(self, *ls):
        random.seed(42)

        if len(ls) == 1:
            li = list(*ls)
            random.shuffle(li)
            return li

        li = list(zip(*ls))
        random.shuffle(li)
        return zip(*li)
    
    def _labels_w(self):
        """VI_HTR dataset reader for words"""
        pt_path = self.source

        paths = {
            "train": open(os.path.join(pt_path, "trainset.txt"), encoding="utf8")
            .read()
            .splitlines(),
            "valid": open(os.path.join(pt_path, "validset.txt"), encoding="utf8")
            .read()
            .splitlines(),
            "test": open(os.path.join(pt_path, "testset.txt"), encoding="utf8")
            .read()
            .splitlines(),
        }

        img_path = os.path.join(self.source, "word")
        dataset = self._init_dataset()

        for data_type, lines in paths.items():
            for line in lines:
                split = line.removesuffix("\n").split("|")
                dataset[data_type]["dt"].append(os.path.join(img_path, f"{split[0]}"))
                dataset[data_type]["gt"].append(split[-1])

        return dataset
    
    def _labels_l(self):
        """VI_HTR dataset reader for lines"""
        pt_path = self.source

        paths = {
            "train": open(os.path.join(pt_path, "trainset.txt"), encoding="utf8")
            .read()
            .splitlines(),
            "valid": open(os.path.join(pt_path, "validset.txt"), encoding="utf8")
            .read()
            .splitlines(),
            "test": open(os.path.join(pt_path, "testset.txt"), encoding="utf8")
            .read()
            .splitlines(),
        }

        img_path = os.path.join(self.source, "line")
        dataset = self._init_dataset()

        for data_type, lines in paths.items():
            for line in lines:
                split = line.removesuffix("\n").split("|")
                dataset[data_type]["dt"].append(os.path.join(img_path, f"{split[0]}"))
                dataset[data_type]["gt"].append(split[-1])

        return dataset
    
    def _labels_l_m(self):
        """VI_HTR dataset reader for lines (modified line extraction function)"""
        pt_path = self.source

        paths = {
            "train": open(os.path.join(pt_path, "trainset.txt"), encoding="utf8")
            .read()
            .splitlines(),
            "valid": open(os.path.join(pt_path, "validset.txt"), encoding="utf8")
            .read()
            .splitlines(),
            "test": open(os.path.join(pt_path, "testset.txt"), encoding="utf8")
            .read()
            .splitlines(),
        }

        img_path = os.path.join(self.source, "line")
        dataset = self._init_dataset()

        for data_type, lines in paths.items():
            for line in lines:
                split = line.removesuffix("\n").split("|")
                dataset[data_type]["dt"].append(os.path.join(img_path, f"{split[0]}"))
                dataset[data_type]["gt"].append(split[-1])

        return dataset
