# Copyright 2018 qyou@nlpr.ia.ac.cn
import os

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('segment')
class Segment(Component):
    def __init__(self,
                 user_dict_path=None,
                 vocab=None,
                 **kwargs):
        self.user_dict_path = user_dict_path
        self.vocab = vocab

    def exists_user_dict(self):
        return self.user_dict_path and os.path.exists(self.user_dict_path)

    def __call__(self, text, *args, **kwargs):
        raise NotImplementedError('You should use the specific word segment class')
