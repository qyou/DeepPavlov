# Copyright 2018 qyou@nlpr.ia.ac.cn
import os

HAS_JIE_BA = False

try:
    import jieba as _J
    import jieba.posseg as _J_P

    HAS_JIE_BA = True
except ImportError:
    pass

from deeppavlov.core.models.component import Component
from deeppavlov.core.data.segment import Segment
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

logger = get_logger(__file__)


class _PackageNotFoundException(Exception):
    pass

@register('jieba_segment')
class JiebaSegment(Segment):
    """Word segment by jieba module: https://github.com/fxsjy/jieba
    """

    def __init__(self,
                 user_dict_path=None,
                 vocab=None,
                 **kwargs):
        super().__init__(user_dict_path,
                         vocab,
                         **kwargs)
        if not HAS_JIE_BA:
            raise _PackageNotFoundException('Please install jieba package using `pip install jieba`')
        self.seg = _J
        self.pseg = _J_P
        if self.exists_user_dict():
            self.seg.load_userdict(user_dict_path)
        if self.vocab:
            for v in self.vocab:
                if isinstance(v, dict):
                    word = v.get('word', None)
                    if word:
                        freq = v.get('freq', None)
                        tag = v.get('tag', None)
                        self.seg.add_word(word, freq, tag)
                elif isinstance(v, str):
                    self.seg.add_word(v)
                else:
                    logger.warning('only string or dictionary is needed! type {} is found!'.format(type(v)))

    def __call__(self, text, *args, **kwargs):
        cut_all = kwargs.get('cut_all', False)
        HMM = kwargs.get('HMM', True)
        tag = kwargs.get('tag', False)
        if tag:
            for word, nature in self.pseg.lcut(text, HMM):
                yield (word, nature)
        else:
            yield from self.seg.cut(text, cut_all, HMM)


    @property
    def raw(self):
        return self.seg

    @property
    def pos_seg(self):
        return self.pseg

