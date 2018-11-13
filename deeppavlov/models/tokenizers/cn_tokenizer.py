# Copyright 2018 qyou@nlpr.ia.ac.cn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Generator, Any, Optional, Union, Tuple

import re
 
# from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.corpus import stopwords
# STOPWORDS = stopwords.words('russian')
# import pymorphy2

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.models.tokenizers.utils import detokenize, ngramize
from deeppavlov.core.common.log import get_logger
from deeppavlov.models.segments.jieba_segment import JiebaSegment


logger = get_logger(__name__)

DEFAULT_ENCODING = 'utf-8'
# default punctuations
DEFAULT_PUNCTUATION = r"""!"#$%&'()*+,-./:;<=>?@[\]^`{|}~、，；：。？！…（）《》"""

@register('cn_tokenizer')
class ChineseTokenizer(Component):
    """Chinese Tokenizer"""
    def __init__(self, segment=None, stopwords=None, encoding=DEFAULT_ENCODING, re_remove_str=DEFAULT_PUNCTUATION, re_split_str=r'\s+', 
    lowercase=True, alphas_only=None, **kwargs):
        self.segment = segment or JiebaSegment()
        self.stopwords = stopwords or []
        self.encoding = encoding
        self.re_split_pattern = re.compile(re_split_str, re.IGNORECASE | re.MULTILINE)
        self.translate_table = str.maketrans(re_remove_str, len(re_remove_str) * ' ')
        self.lowercase = lowercase
        self.alphas_only = alphas_only
    
    def check_valid(self, text):
        if isinstance(text, bytes):
            text = text.decode(self.encoding)
        if not isinstance(text, str):
            logger.warning('need str|bytes input, but find {}'.format(type(text)))
            return False
        return True    

    def __call__(self, text: Union[Union[str, bytes], List[Union[str, bytes]]], *args, **kwargs):
        if isinstance(text, (str, bytes)):
            if self.check_valid(text):
                text = text.translate(self.translate_table)
                items = self.re_split_pattern.split(text)
                return list(self._segment(items))
            else:
                raise ValueError('Value error, str|bytes or str|bytes list is needed')
        else:
            tokens_list = []
            for item in text:
                if self.check_valid(item):
                    item = item.translate(self.translate_table)
                    items = self.re_split_pattern.split(item)
                    tokens_list.append(list(self._segment(items)))
                else:
                    continue
            return tokens_list
    
    def _segment(self, docs, tag=False, lowercase = True):
        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        if isinstance(docs, str):
            tokens = list(self.segment(docs))
            if _lowercase:
                tokens = [t.lower() for t in tokens]
            yield from self._filter(tokens)
        else:
            for i, doc in enumerate(docs):
                # logger.info("Tokenize doc {} from {}".format(i, size))
                tokens = list(self.segment(doc))
                if _lowercase:
                    tokens = [t.lower() for t in tokens]
                yield from self._filter(tokens)

    def _filter(self, items: List[str], alphas_only=True):
        """Filter a list of tokens.

        Args:
            items: a list of tokens to filter
            alphas_only: whether to filter out non-alpha tokens

        Returns:
            a list of filtered tokens

        """
        if self.alphas_only is None:
            _alphas_only = alphas_only
        else:
            _alphas_only = self.alphas_only

        if _alphas_only:
            filter_fn = lambda x: x.isalpha() and not x.isspace() and x not in self.stopwords
        else:
            filter_fn = lambda x: not x.isspace() and x not in self.stopwords

        for t in filter(filter_fn, items):
            yield t

    def add_stopword(self, word):
        if word not in self.stopwords:
            self.stopwords.append(word)


    def set_stopword(self, stopwords):
        """Filter a list of tokens.

        Args:
            items: a list of tokens to filter
            alphas_only: whether to filter out non-alpha tokens

        Returns:
            a list of filtered tokens

        """
        self.stopwords = stopwords



def main():
    tokenizer = ChineseTokenizer()
    print(tokenizer(['学习python， 当上程序员，迎娶白富美，走上人生巅峰！','根据需求，我们买了二十四口交换机用于网络设备的更新升级']))


if __name__ == '__main__':
    main()