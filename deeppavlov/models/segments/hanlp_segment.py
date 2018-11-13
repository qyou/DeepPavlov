# Copyright 2018 qyou@nlpr.ia.ac.cn
import os

HAS_HAN_LP = False
try:
    import pyhanlp as _H

    HAS_HAN_LP = True
except ImportError:
    pass

from deeppavlov.core.models.component import Component
from deeppavlov.core.data.segment import Segment
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

logger = get_logger(__file__)


class _PackageNotFoundException(Exception):
    pass

@register('hanlp_segment')
class HanLPSegment(Segment):
    """Word segment by pyhanlp module: https://github.com/hankcs/pyhanlp
    """

    def __init__(self,
                 user_dict_path=None,
                 vocab=None,
                 **kwargs):
        super().__init__(user_dict_path, vocab, **kwargs)
        if not HAS_HAN_LP:
            raise _PackageNotFoundException('Please install pyhanlp package using `pip install pyhanlp`')
        self.seg = _H.HanLP
        self.dictionary = _H.CustomDictionary
        if self.exists_user_dict():
            with open(self.user_dict_path, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if line is None or line == '' or line.startswith('#'):
                        logger.warning('pass the blank line')
                    else:
                        words = line.split(maxsplit=1)
                        if len(words) == 1:
                            self.dictionary.add(words[0])
                        elif len(words) == 2:
                            self.dictionary.add(words[0], words[1])
                        else:
                            logger.warning('cannot split the line to word or word and tag')
        if self.vocab:
            for v in self.vocab:
                if isinstance(v, dict):
                    word = v.get('word', None)
                    if word:
                        freq = v.get('freq', None)
                        tag = v.get('tag', None)
                        tag_freq_str = ' '.join([e for e in (freq, tag) if e is not None])
                        if tag_freq_str.strip() != '':
                            self.dictionary.add(word, tag_freq_str)
                        else:
                            self.dictionary.add(word)
                elif isinstance(v, str):
                    self.dictionary.add(v)
                else:
                    logger.warning('only string or dictionary is needed! type {} is found!'.format(type(v)))

    def __call__(self, text, *args, **kwargs):
        tag = kwargs.get('tag', False)
        for w in self.seg.segment(text):
            if not tag:
                yield w.word
            else:
                yield (w.word, str(w.nature))

    @property
    def raw(self):
        return self.seg

    @property
    def dic(self):
        return self.dictionary
