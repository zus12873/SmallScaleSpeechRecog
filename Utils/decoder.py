from Utils.Tokenizer import TextProcess  #Utils.
import ctcdecode
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
    # "<s>",
    # "</s>",
    # "<unk>"
]


# labels ={"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}

def DecodeGreedy(output, blank_label=labels.index("<PAD>"), collapse_repeated=True): 
    arg_maxes = torch.argmax(output, dim=2).squeeze(0)
    decode = []  
    # print(arg_maxes.size())
    for i, index in enumerate(arg_maxes):
        # print(index)
        if index != blank_label:
            if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
                continue
            decode.append(index.item())
    return textprocess.int_to_text_sequence(decode)

class CTCBeamDecoder:

    def __init__(self, beam_size=50, blank_id=labels.index("<PAD>"), kenlm_path=None ):
        print("loading beam search with lm...")
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels, alpha=0.4, beta=0.85,
            beam_width=beam_size, blank_id=labels.index("<PAD>"),
            model_path=kenlm_path)
        print("finished loading beam search")

    def __call__(self, output):
        # print("try")
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])
