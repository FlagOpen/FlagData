from ..base_operator import BaseOperator
from .email_pruner import EmailPruner
from .link_pruner import LinkPruner
from .repeat_sentence_pruner import RepeatSentencePruner
from .table_pruner import TablePruner
from .copyright_pruner import CopyrightPruner
from .ip_pruner import IpPruner
from .non_chinese_char_pruner import NonChineseCharPruner
from .replace_pruner import ReplacePruner
from .unicode_pruner import UnicodePruner
from .figuret_able_caption_pruner import FigureTableCaptionPruner

__all__ = ['EmailPruner', 'LinkPruner', 'RepeatSentencePruner', 'TablePruner', 'CopyrightPruner',
           'IpPruner', 'NonChineseCharPruner', 'ReplacePruner', 'UnicodePruner', 'FigureTableCaptionPruner']
