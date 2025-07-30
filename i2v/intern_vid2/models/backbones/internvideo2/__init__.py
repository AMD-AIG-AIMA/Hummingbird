# download from https://github.com/Ji4chenLi/t2v-turbo/tree/main/intern_vid2/models/backbones/internvideo2, whcih is modified from https://github.com/AILab-CVC/VideoCrafter
from .internvl_clip_vision import internvl_clip_6b
from .internvideo2 import pretrain_internvideo2_1b_patch14_224, pretrain_internvideo2_6b_patch14_224
from .internvideo2_clip_vision import InternVideo2
from .internvideo2_clip_text import LLaMA, Tokenizer