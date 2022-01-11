# coding: utf-8

class ImageDictionary:
    """a wrapper for Image Dictionary"""
    def __init__(self, vocab_size, bos="<s>", pad="<pad>", eos="</s>", unk="<unk>", extra_special_symbols=None):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos

        self.symbols = [i for i in range(vocab_size)]
        self.count = [1 for i in range(vocab_size)]
        self.indices = {i: i for i in range(vocab_size)}

        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)

        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def get_count(self, idx):
        return self.count[idx]

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index