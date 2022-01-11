import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from fairseq import utils
from fairseq.data import iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from data.t2i_dataset import T2IDataset
from data.img_dictionary import ImageDictionary
from utils.wordpiece import BertTokenizer

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

@dataclass
class T2IConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    dict_file: Optional[str] = field(
        default=None, metadata={"help": "chinese dict file"}
    )
    vocab_file: Optional[str] = field(
        default=None, metadata={"help": "chinese vocab file"}
    )
    caption_path: Optional[str] = field(
        default=None, metadata={"help": "caption data path"}
    )
    image_path: Optional[str] = field(
        default=None, metadata={"help": "image data path"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1025, metadata={"help": "max number of tokens in the target sequence"}
    )  # +1 for [eos]

    image_size: int = field(
        default=256, metadata={"help": "image size"}
    )
    image_vocab_size: int = field(
        default=8192, metadata={"help": "image vocab size"}
    )
    max_src_length: int = field(
        default=40, metadata={"help": "the maximum source sequence length"}
    )
    max_tgt_length: int = field(
        default=2, metadata={"help": "the maximum target sequence length"}
    )


@register_task("t2i", dataclass=T2IConfig)
class T2ITask(FairseqTask):

    cfg: T2IConfig

    def __init__(self, cfg: T2IConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: T2IConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        src_dict = cls.load_dictionary(cfg.dict_file)
        tgt_dict = ImageDictionary(cfg.image_vocab_size)

        logger.info("src dictionary: {} types".format(len(src_dict)))
        logger.info("src dictionary: {} types".format(len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        data_path = self.cfg.data

        path_split = split
        if split.startswith('val'):
            path_split = 'val'
            logger.info(f"mapping split [val] to [{split}]")

        caption_path = os.path.join(data_path, 'T2I_{}.text.tsv'.format(path_split)) if self.cfg.caption_path is None else self.cfg.caption_path
        image_path = os.path.join(data_path, 'T2I_{}.img.tsv'.format(path_split)) if self.cfg.image_path is None else self.cfg.image_path

        self.datasets[split] = T2IDataset(
            caption_path,
            image_path,
            self.tokenizer,
            self.src_dict,
            self.tgt_dict,
            left_pad_source=self.cfg.left_pad_source,  # left pad or right pad
            left_pad_target=self.cfg.left_pad_target,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,  # not used
            image_size=self.cfg.image_size,
        )

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):

        # return a reusable, sharded iterator
        epoch_iter = iterators.StreamingEpochBatchIterator(
            dataset=dataset,
            max_sentences=max_sentences,
            collate_fn=dataset.collater,
            epoch=epoch,
            num_workers=num_workers,
            buffer_size=data_buffer_size,
            timeout=0
        )

        return epoch_iter

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.tokenizer = BertTokenizer(cfg.vocab_file)
        return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict