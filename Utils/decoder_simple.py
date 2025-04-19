from Utils.Tokenizer import TextProcess
import torch

textprocess = TextProcess()

labels = [
    "<PAD>",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z", 
    "'",  # 28
    "<UNK>",  # 29, blank
]

def DecodeGreedy(output, blank_label=labels.index("<PAD>"), collapse_repeated=True): 
    arg_maxes = torch.argmax(output, dim=2).squeeze(0)
    decode = []  
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
                continue
            decode.append(index.item())
    return textprocess.int_to_text_sequence(decode)

class CTCBeamDecoder:
    """简化版的CTCBeamDecoder，仅使用贪婪解码，不依赖ctcdecode包"""

    def __init__(self, beam_size=50, blank_id=labels.index("<PAD>"), kenlm_path=None):
        print("注意：使用简化版解码器，没有使用语言模型和波束搜索")
        self.blank_id = blank_id

    def __call__(self, output):
        # 使用贪婪解码代替波束搜索
        return DecodeGreedy(output, blank_label=self.blank_id)

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]]) 