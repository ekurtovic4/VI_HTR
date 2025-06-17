"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

import h5py
import numpy as np
import data.preproc as pp


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, source, batch_size, charset, max_text_length, max_time_steps=256, predict=False, stream=False):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size
        self.max_time_steps = max_time_steps

        if max_text_length > max_time_steps:
            print(f"Warning: max_text_length ({max_text_length}) > max_time_steps ({max_time_steps}). "
                  f"This will cause CTC errors. Setting max_text_length = max_time_steps.")
            self.tokenizer.maxlen = max_time_steps

        self.size = dict()
        self.steps = dict()
        self.index = dict()

        if stream:
            self.dataset = h5py.File(source, "r")

            for pt in ['train', 'valid', 'test']:
                self.size[pt] = self.dataset[pt]['gt'][:].shape[0]
                self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
        else:
            self.dataset = dict()

            with h5py.File(source, "r") as f:
                for pt in ['train', 'valid', 'test']:
                    self.dataset[pt] = dict()
                    self.dataset[pt]['dt'] = np.array(f[pt]['dt'])
                    self.dataset[pt]['gt'] = np.array(f[pt]['gt'])

                    self.size[pt] = len(self.dataset[pt]['gt'])
                    self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))

        self.stream = stream
        self.arange = np.arange(len(self.dataset['train']['gt']))
        np.random.seed(42)

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        self.index['train'] = 0

        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

                if not self.stream:
                    np.random.shuffle(self.arange)
                    self.dataset['train']['dt'] = self.dataset['train']['dt'][self.arange]
                    self.dataset['train']['gt'] = self.dataset['train']['gt'][self.arange]

            index = self.index['train']
            until = index + self.batch_size
            self.index['train'] = until

            x_train = self.dataset['train']['dt'][index:until]
            x_train = pp.augmentation(x_train,
                                      rotation_range=1.5,
                                      scale_range=0.05,
                                      height_shift_range=0.025,
                                      width_shift_range=0.05)
            x_train = pp.normalization(x_train)

            y_train_raw = self.dataset['train']['gt'][index:until]

            # Check max label length in this batch
            max_label_len = max(len(y) for y in y_train_raw)
            if max_label_len > self.tokenizer.maxlen:
                print(f"Warning: Found label length {max_label_len} > tokenizer maxlen {self.tokenizer.maxlen}. "
                      "Labels will be truncated, which can cause errors with CTC loss.")

            # Encode and pad labels (truncate if too long)
            y_train = []
            for y in y_train_raw:
                encoded = self.tokenizer.encode(y)
                if len(encoded) > self.tokenizer.maxlen:
                    encoded = encoded[:self.tokenizer.maxlen]  # truncate to maxlen
                padded = np.pad(encoded, (0, self.tokenizer.maxlen - len(encoded)))
                y_train.append(padded)

            y_train = np.asarray(y_train, dtype=np.int16)

            yield (x_train, y_train)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        self.index['valid'] = 0

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = index + self.batch_size
            self.index['valid'] = until

            x_valid = self.dataset['valid']['dt'][index:until]
            x_valid = pp.normalization(x_valid)

            y_valid_raw = self.dataset['valid']['gt'][index:until]

            # Check max label length
            max_label_len = max(len(y) for y in y_valid_raw)
            if max_label_len > self.tokenizer.maxlen:
                print(f"Warning: Found validation label length {max_label_len} > tokenizer maxlen {self.tokenizer.maxlen}. "
                      "Labels will be truncated.")

            y_valid = []
            for y in y_valid_raw:
                encoded = self.tokenizer.encode(y)
                if len(encoded) > self.tokenizer.maxlen:
                    encoded = encoded[:self.tokenizer.maxlen]
                padded = np.pad(encoded, (0, self.tokenizer.maxlen - len(encoded)))
                y_valid.append(padded)

            y_valid = np.asarray(y_valid, dtype=np.int16)

            yield (x_valid, y_valid)

    def next_test_batch(self):
        """Return model predict parameters"""

        self.index['test'] = 0

        while True:
            if self.index['test'] >= self.size['test']:
                self.index['test'] = 0
                break

            index = self.index['test']
            until = index + self.batch_size
            self.index['test'] = until

            x_test = self.dataset['test']['dt'][index:until]
            x_test = pp.normalization(x_test)

            yield (x_test,)


class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        if isinstance(text, bytes):
            text = text.decode()

        encoded = []
        for item in " ".join(text.split()):
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
