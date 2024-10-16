""""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""
"""Build Datasets for Hansards

Don't go publishing results based on this. We restrict a lot of things to make
this nicer.
"""

import locale
import os
import re
from string import punctuation
from collections import Counter, OrderedDict
from collections.abc import Sequence, Iterator
from typing import Optional, Union, IO
import gzip
import torch


TOKENIZER_PATTERN = re.compile(r"[" + re.escape(punctuation) + r"\d\s]+")

locale.setlocale(locale.LC_ALL, "C")  # ensure reproducible sorting

__all__ = [
    "get_dir_lines",
    "build_vocab_from_dir",
    "word2id_to_id2word",
    "id2word_to_word2id",
    "write_stoi_to_file",
    "read_stoi_from_file",
    "get_common_prefixes",
    "HansardDataset",
    "HansardDataLoader",
    "HansardEmptyDataset",
]


def open_path(mode):
    def decorator(func):
        def wrapper(*args, **kwargs):
            path, *rest = args
            if path.suffix == ".gz":
                open_ = gzip.open
            else:
                open_ = open

            with open_(path, mode=mode) as open_file:
                return func(open_file, *rest, **kwargs)

        return wrapper

    return decorator


def get_special_symbols(word2id: dict) -> tuple:
    _sos = word2id.get("<s>")
    _eos = word2id.get("</s>")
    _pad = word2id.get("<blank>")
    _unk = word2id.get("<unk>")
    return _sos, _eos, _pad, _unk


def get_dir_lines(dir_: str, lang: str, filenames: Sequence[str] = None) -> None:
    """Generate line info from data in a directory for a given language

    Parameters
    ----------
    dir_ : str
        A path to the transcription directory.
    lang : {'e', 'f'}
        Whether to tokenize the English sentences ('e') or French ('f').
    filenames : sequence, optional
        Only tokenize sentences with matching names. If :obj:`None`, searches
        the whole directory in C-sorted order.

    Yields
    ------
    tokenized, filename, offs : list
        `tokenized` is a list of tokens for a line. `filename` is the source
        file. `offs` is the start of the sentence in the file, to seek to.
        Lines are yielded by iterating over lines in each file in the order
        presented in `filenames`.
    """
    _in_set_check("lang", lang, {"e", "f"})
    lang = "." + lang
    if filenames is None:
        filenames = sorted(os.listdir(dir_))
    for filename in filenames:
        if filename.endswith(lang):
            with open(os.path.join(dir_, filename)) as f:
                offs = f.tell()
                line = f.readline()
                while line:
                    yield (
                        [w for w in TOKENIZER_PATTERN.split(line.lower()) if w],
                        filename,
                        offs,
                    )
                    offs = f.tell()
                    line = f.readline()


def build_vocab_from_dir(
    train_dir_: str,
    lang: str,
    max_vocab: int = 20000,
    min_freq: int = 1,
    specials: Optional[list[str]] = [],
) -> dict:
    """Build a vocabulary (words->ids) from transcriptions in a directory

    Parameters
    ----------
    train_dir_ : str
        A path to the transcription directory. ALWAYS use the training
        directory, not the test, directory, when building a vocabulary.
    lang : {'e', 'f'}
        Whether to build the English vocabulary ('e') or the French one ('f').
    max_vocab : int, optional
        The size of your vocabulary. Words with the greatest count will be
        retained.
    min_freq: The minimum frequency needed to include a token in the vocabulary.
    specials: Special symbols to add. The order of supplied tokens will be preserved.

    Returns
    -------
    word2id : dict
        A dictionary of keys being words, values being ids. There will be an
        entry for each id between ``[0, max_vocab - 1]`` inclusive.
    """
    _in_range_check("max_vocab", max_vocab, -1)
    word2count = Counter()
    for tokenized, _, _ in get_dir_lines(train_dir_, lang):
        word2count.update(tokenized)
    word2count = sorted(word2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if max_vocab:
        word2count = word2count[: max_vocab - len(specials)]
    word2count = OrderedDict(word2count)
    specials = specials or []
    for symbol in specials:
        word2count.pop(symbol, None)
    tokens = []
    for token, freq in word2count.items():
        if freq >= min_freq:
            tokens.append(token)
    tokens[0:0] = specials
    return dict((v, i) for i, v in enumerate(tokens))


def word2id_to_id2word(word2id: dict) -> dict:
    """word2id -> id2word"""
    return dict((v, k) for (k, v) in word2id.items())


def id2word_to_word2id(id2word: dict) -> dict:
    """id2word -> word2id"""
    return dict((v, k) for (k, v) in id2word.items())


@open_path("wt")
def write_stoi_to_file(file_: IO, word2id: dict) -> None:
    """Write string to id (stoi) or a word2id map to a file

    Parameters
    ----------
    file_ : str or file
        A file to write `word2id` to. If a path that ends with ``.gz``, it will
        be gzipped.
    word2id : dict
        A dictionary of keys being words, values being ids
    """
    id2word = word2id_to_id2word(word2id)
    for i in range(len(id2word)):
        file_.write("{} {}\n".format(id2word[i], i))


@open_path("rt")
def read_stoi_from_file(file_: IO) -> dict:
    """Read string to id (stoi) or a word2id map from a file

    Parameters
    ----------
    file_ : str or file
        A file to read `word2id` from. If a path that ends with ``.gz``, it
        will be de-compressed via gzip.

    Returns
    -------
    word2id : dict
        A dictionary of keys being words, values being ids
    """
    ids = set()
    word2id = dict()
    for line in file_:
        line = line.strip()
        if not line:
            continue
        word, id_ = line.split()
        id_ = int(id_)
        if id_ in ids:
            raise ValueError(f"Duplicate id {id_}")
        if word in word2id:
            raise ValueError(f"Duplicate word {word}")
        ids.add(id_)
        word2id[word] = id_
    _word2id_validity_check("word2id", word2id)
    return word2id


def get_common_prefixes(dir_: str) -> Sequence[str]:
    """Return a list of file name prefixes common to both English and French

    A prefix is common to both English and French if the files
    ``<dir_>/<prefix>.e`` and ``<dir_>/<prefix>.f`` both exist.

    Parameters
    ----------
    dir_ : str
        A path to the transcription directory.

    Returns
    -------
    common : list
        A C-sorted list of common prefixes
    """
    all_fns = os.listdir(dir_)
    english_fns = set(fn[:-2] for fn in all_fns if fn.endswith(".e"))
    french_fns = set(fn[:-2] for fn in all_fns if fn.endswith(".f"))
    del all_fns
    common = english_fns & french_fns
    if not common:
        raise ValueError(
            f"Directory {dir_} contains no common files ending in .e or "
            f".f. Are you sure this is the right directory?"
        )
    return sorted(common)


class HansardDataset(torch.utils.data.Dataset):
    """A dataset of a partition of the Canadian Hansards

    Indexes bitext sentence pairs ``source_x, target_y``, where ``source_x`` is the source language
    sequence and ``target_y`` is the corresponding target language sequence.

    Parameters
    ----------
    dir_ : str
        A path to the data directory
    french_word2id : dict or str
        Either a dictionary of French words to ids, or a path pointing to one.
    english_word2id : dict or str
        Either a dictionary of English words to ids, or a path pointing to one.
    source_language : {'e', 'f'}, optional
        Specify the language we're translating from. By default, it's French
        ('f'). In the case of English ('e'), ``source_x`` is still the source language
        sequence, but it now refers to English.
    prefixes : sequence, optional
        A list of file prefixes in `dir_` to consider part of the dataset. If
        :obj:`None`, will search for all common prefixes in the directory.

    Attributes
    ----------
    dir_ : str
    source_language : {'e', 'f'}
    source_unk_id : int
        A special id to indicate a source token was out-of-vocabulary.
    source_pad_id : int
        A special id used for right-padding source-sequences during batching
    source_vocab_size : int
        The total number of unique ids in source sequences. All ids are bound
        between ``[0, source_vocab_size - 1]`` inclusive. Includes
        `source_unk_id` and `source_pad_id`.
    target_unk_id : int
        A special id to indicate a target token was in-vocabulary.
    target_sos_id : int
        A special id to indicate the start of a target sequence. One SOS token
        is prepended to each target sequence ``target_y``.
    target_eos_id : int
        A special id to indicate the end of a target sequence. One EOS token
        is appended to each target sequence ``target_y``.
    target_vocab_size : int
        The total number of unique ids in target sequences. All ids are bound
        between ``[0, target_vocab_size - 1]`` inclusive. Includes
        `target_unk_id`, `target_sos_id`, and `target_eos_id`.
    pairs : tuple
    """

    def __init__(
        self,
        dir_: str,
        french_word2id: Union[dict, str],
        english_word2id: Union[dict, str],
        source_language: str = "f",
        prefixes: Sequence[str] = None,
    ):
        _in_set_check("source_language", source_language, {"e", "f"})
        if isinstance(french_word2id, str):
            french_word2id = read_stoi_from_file(french_word2id)
        else:
            _word2id_validity_check("french_word2id", french_word2id)
        if isinstance(english_word2id, str):
            english_word2id = read_stoi_from_file(english_word2id)
        else:
            _word2id_validity_check("english_word2id", english_word2id)
        if prefixes is None:
            prefixes = get_common_prefixes(dir_)
        english_fns = (p + ".e" for p in prefixes)
        french_fns = (p + ".f" for p in prefixes)
        english_l = get_dir_lines(dir_, "e", english_fns)
        french_l = get_dir_lines(dir_, "f", french_fns)
        if source_language == "f":
            source_word2id = french_word2id
            target_word2id = english_word2id
        else:
            source_word2id = english_word2id
            target_word2id = french_word2id
        pairs = []

        (
            source_sos_id,
            source_eos_id,
            source_pad_id,
            source_unk_id,
        ) = get_special_symbols(source_word2id)
        target_sos_id, target_eos_id, target_pad, target_unk_id = get_special_symbols(
            target_word2id
        )

        for (e, e_fn, _), (f, f_fn, _) in zip(english_l, french_l):
            assert e_fn[:-2] == f_fn[:-2]
            if not e or not f:
                assert not e and not f  # if either is empty, both should be
                continue
            if source_language == "f":
                source_x, target_y = f, e
            else:
                source_x, target_y = e, f
            source_x = torch.tensor(
                [source_word2id.get(w, source_unk_id) for w in source_x]
            )
            # source_x = torch.tensor(
            #     [source_sos_id] + [source_word2id.get(w, source_unk_id) for w in source_x] + [source_eos_id])
            target_y = torch.tensor(
                [target_sos_id]
                + [target_word2id.get(w, target_unk_id) for w in target_y]
                + [target_eos_id]
            )
            if torch.all(source_x == source_unk_id) and torch.all(
                target_y[1:-1] == target_unk_id
            ):
                # skip sentences that are solely OOV
                continue
            pairs.append((source_x, target_y))
        self.dir_ = dir_
        self.source_language = source_language
        self.source_vocab_size = len(source_word2id)
        self.source_unk_id = source_unk_id
        self.source_pad_id = source_pad_id
        self.target_unk_id = target_unk_id
        self.target_sos_id = target_sos_id
        self.target_eos_id = target_eos_id
        self.target_vocab_size = len(target_word2id)
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


class HansardEmptyDataset(HansardDataset):
    """A dummy dataset that only keeps the vocabulary and meta information.

    Consult :class:`HansardDataset` for a description of parameters and
    attributes
    """

    def __init__(
        self,
        french_word2id: Union[dict, str],
        english_word2id: Union[dict, str],
        source_language: str = "f",
        prefixes: Sequence[str] = None,
    ):
        _in_set_check("source_language", source_language, {"e", "f"})
        if isinstance(french_word2id, str):
            french_word2id = read_stoi_from_file(french_word2id)
        else:
            _word2id_validity_check("french_word2id", french_word2id)
        if isinstance(english_word2id, str):
            english_word2id = read_stoi_from_file(english_word2id)
        else:
            _word2id_validity_check("english_word2id", english_word2id)

        if source_language == "f":
            source_word2id = french_word2id
            target_word2id = english_word2id
        else:
            source_word2id = english_word2id
            target_word2id = french_word2id

        (
            source_sos_id,
            source_eos_id,
            source_pad_id,
            source_unk_id,
        ) = get_special_symbols(source_word2id)
        (
            target_sos_id,
            target_eos_id,
            target_pad_id,
            target_unk_id,
        ) = get_special_symbols(target_word2id)

        self.source_language = source_language
        self.source_vocab_size = len(source_word2id)
        self.source_sos_id, self.source_eos_id = (
            source_sos_id,
            source_eos_id,
        )
        self.source_unk_id = source_unk_id
        self.source_pad_id = source_pad_id
        self.target_unk_id = target_unk_id
        self.target_sos_id = target_sos_id
        self.target_eos_id = target_eos_id
        self.target_vocab_size = len(target_word2id)
        self.source_word2id = source_word2id
        self.target_id2word = word2id_to_id2word(target_word2id)

    def __len__(self) -> int:
        return ValueError("This is a placeholder dataset. No actual data is loaded.")

    def __getitem__(self, i: int) -> tuple[str, str]:
        return ValueError("This is a placeholder dataset. No actual data is loaded.")

    def tokenize(self, sentence: str) -> list[str]:
        """Tokenize the given sentence.

        Parameters
        ----------
        sentence: str
            The sentence to be tokenized.

        Returns
        -------
        tokenized: list[str]
            The tokenized sentence.
        """
        tokenized = [x for x in TOKENIZER_PATTERN.split(sentence.lower()) if x]
        return tokenized


class LengthBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
    Allows us to sort the data by sequence length for the purpose of making smaller batches in the sequence dimension.
    We just add src and tgt lengths to the data and sort by the sum of the two.  This should work good enough for a
    french-english translation task.
    """

    def __init__(self, data: HansardDataset, batch_size: int) -> None:
        self.data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        sizes = torch.tensor([len(x[0]) + len(x[1]) for x in self.data.pairs])
        for batch in torch.chunk(torch.argsort(sizes), len(self)):
            yield batch.tolist()


class HansardDataLoader(torch.utils.data.DataLoader):
    """A DataLoader yielding batches of bitext

    Consult :class:`HansardDataset` for a description of parameters and
    attributes

    Parameters
    ----------
    dir_ : str
    french_word2id : dict or str
    english_word2id : dict or str
    source_language : {'e', 'f'}, optional
    prefixes : sequence, optional
    kwargs : optional
        See :class:`torch.utils.data.DataLoader` for additional arguments.
        Do not specify `collate_fn`.
    """

    def __init__(
        self,
        dir_: str,
        french_word2id: Union[dict, str],
        english_word2id: Union[dict, str],
        source_language: str = "f",
        prefixes: Sequence[str] = None,
        arch_type: str = "seq2seq",
        is_distributed: bool = False,
        shuffle: bool = False,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            raise TypeError(
                "HansardDataLoader() got an unexpected keyword argument " "'collate_fn'"
            )
        dataset = HansardDataset(
            dir_, french_word2id, english_word2id, source_language, prefixes
        )

        _collate_fn = (
            self.rnn_collate if arch_type == "seq2seq" else self.transformer_collate
        )

        batch_size = kwargs.get("batch_size", 1)
        #  batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        bad = {"batch_size", "shuffle", "sampler", "drop_last"}
        kwargs = {k: v for k, v in kwargs.items() if k not in bad}

        if shuffle:  # important for training the transformer model
            # print('Using SubsetRandomSampler')
            _sampler = torch.utils.data.SubsetRandomSampler(list(range(len(dataset))))
            super().__init__(
                dataset,
                collate_fn=_collate_fn,
                sampler=_sampler,
                batch_size=batch_size,
                **kwargs,
            )
        else:
            # print('Using LengthBatchSampler')
            _sampler = LengthBatchSampler(dataset, batch_size)
            super().__init__(
                dataset, collate_fn=_collate_fn, batch_sampler=_sampler, **kwargs
            )

        self.max_padding = kwargs.get("max_padding", 128)
        self.is_testing = kwargs.get("test", prefixes == None)

        # 0, 1, 2, 3
        (
            self.target_sos_id,
            self.target_eos_id,
            self.target_pad_id,
            self.target_unk_id,
        ) = get_special_symbols(english_word2id)
        (
            self.source_sos_id,
            self.source_eos_id,
            self.source_pad_id,
            self.source_unk_id,
        ) = get_special_symbols(french_word2id)
        self.pad_id = kwargs.get("pad_id", 2)  # default itos(2) -> "<blank>"

    def rnn_collate(self, seq):
        source_x, target_y = zip(*seq)
        source_lens = torch.tensor([len(f) for f in source_x])
        source_x = torch.nn.utils.rnn.pad_sequence(
            source_x, padding_value=self.dataset.source_pad_id
        )
        target_y = torch.nn.utils.rnn.pad_sequence(
            target_y, padding_value=self.dataset.target_eos_id
        )
        return source_x, source_lens, target_y

    def transformer_collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        batch: list of batch_size tuples of (src, tgt) where src and tgt 1-d tensors
            where the tgts have been padded with <s> and </s> tokens but not srcs.
        return: src [batch_size, max_src_len], src_lens [batch_size], tgt [batch_size, max_tgt_len]
        """

        src_sos = torch.tensor([self.source_sos_id])
        src_eos = torch.tensor([self.source_eos_id])  # </s> token id

        max_src_len = min(
            max(len(src) + 2 for (src, _) in batch), self.max_padding
        )  # +2 for sos and eos
        max_tgt_len = min(max(len(tgt) for (_, tgt) in batch), self.max_padding)

        src_list, src_lens, tgt_list = [], [], []
        for _src, _tgt in batch:
            src_lens.append(torch.tensor([len(_src)]))
            src = torch.cat([src_sos, _src, src_eos], dim=0)[
                : self.max_padding
            ]  # pad and cut to max_padding
            tgt = _tgt[: self.max_padding]  # cut to max_padding
            src_list.append(
                torch.nn.functional.pad(
                    src, pad=(0, max_src_len - len(src)), value=self.pad_id
                )
            )
            tgt_list.append(
                torch.nn.functional.pad(
                    tgt, pad=(0, max_tgt_len - len(tgt)), value=self.pad_id
                )
            )
        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        src_lens = torch.stack(src_lens)
        return src, src_lens, tgt


def _in_range_check(
    name: str,
    value: int,
    low: Union[int, float] = -float("inf"),
    high: Union[int, float] = float("inf"),
    error: Exception = type[ValueError],
):
    if value < low:
        raise error(f"{name} ({value}) is less than {low}")
    if value > high:
        raise error(f"{name} ({value}) is greater than {high}")


def _in_set_check(
    name: str, value: int, set_: str, error: type[Exception] = ValueError
):
    if value not in set_:
        raise error(f"{name} not in {set_}")


def _word2id_validity_check(
    name: str, word2id: dict, error: type[Exception] = ValueError
):
    if set(word2id.values()) != set(range(len(word2id))):
        raise error(
            f"Ids in {name} should be contiguous and span [0, len({name}) - 1]"
            f" inclusive"
        )
