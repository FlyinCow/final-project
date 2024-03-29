from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn as nn

# todo: 测试英文数据集


class ConllDataset(Dataset):
    """
    从conll格式的数据集得到的依存关系数据集。
    """

    def __init__(self, data_path: str):
        self.sentencese = []
        self.labels = []
        lines = (x for x in open(data_path))
        for buf in tqdm(
            self.generate_lines_for_sent(lines), desc="[processing conll data]"
        ):
            conll_lines = []
            for line in buf:
                conll_lines.append(line.strip().split("\t"))

            self.sentencese.append(
                self.generate_sentence_obj_from_conll_lines(conll_lines)
            )
            # sentence_obj = self.sentence_class(*zip(*conll_lines))
            # self.sentencese.append(sentence_obj)

    def generate_lines_for_sent(self, lines):
        """
        将conll格式属于同一个句子的多行组合成一个object
        """
        buf = []
        for line in lines:
            if line.startswith("#"):
                continue
            if not line.strip():
                if buf:
                    yield buf
                    buf = []
                else:
                    continue
            else:
                buf.append(line.strip())
        if buf:
            yield buf

    def generate_sentence_obj_from_conll_lines(self, conll_lines):
        sent = [*zip(*conll_lines)]
        raw_sentence = "".join(sent[1])
        word_count = len(sent[1])
        char_count = len(raw_sentence)
        head_indexs = []
        raw_words = []
        word_pos = []
        word_len = []
        # depths = []
        pos = 0
        for i, line in enumerate(conll_lines):
            # 0 index
            # 1 sentence
            # 2 lemma_sentence
            # 3 upos_sentence
            # 4 xpos_sentence
            # 5 morph
            # 6 head_indices
            # 7 governance_relations
            # 8 secondary_relations
            # 9 extra_info
            raw_word = line[1]
            head_indice = int(line[6])
            head_indexs.append(head_indice - 1)
            if head_indice == 0:
                root_idx = i
            # depths.append(0 if head_indice == 0 else -1)
            raw_words.append(raw_word)
            word_pos.append(pos)
            word_len.append(len(raw_word))
            pos += len(raw_word)

        # for i in range(len(raw_words)):
        #     update_depth(root_idx, i, head_indexs, depths)
        # return (
        #     raw_sentence,
        #     len(words),
        #     root_idx,
        #     words,
        #     head_indexs,
        #     depths,
        # )

        return {
            "raw_sentence": raw_sentence,
            "words_count": word_count,
            "char_count": char_count,
            "root_idx": root_idx,
            "raw_words": raw_words,
            "word_pos": word_pos,
            "word_len": word_len,
            "head_indexs": head_indexs,
            # "depths": depths,
        }

    def set_labels(self, task):
        self.labels.clear()
        for sentence in tqdm(self.sentencese, desc="[computing labels]"):
            self.labels.append(task.labels(sentence))
        return self

    def __len__(self):
        return len(self.sentencese)

    def __getitem__(self, index):
        if len(self.labels) == 0:
            raise ValueError(
                "Call `ConllDataset.sel_labels` first to set labels according to task."
            )
        return self.sentencese[index], self.labels[index]


# note: collate_fn check
def padding_collate_fn(batch_sentence):
    # batch_sentence:[(sentence,label),...]
    words = [x[0]["raw_words"] for x in batch_sentence]
    word_pos = [torch.tensor(x[0]["word_pos"], device="cuda") for x in batch_sentence]
    word_len = [torch.tensor(x[0]["word_len"], device="cuda") for x in batch_sentence]
    word_pos = nn.utils.rnn.pad_sequence(word_pos, batch_first=True, padding_value=-1)
    word_len = nn.utils.rnn.pad_sequence(word_len, batch_first=True, padding_value=1)
    sentences = [x[0]["raw_sentence"] for x in batch_sentence]

    # 手动pad labels
    label_shape = batch_sentence[0][1].shape
    lengths = [torch.tensor(x[0]["words_count"], device="cuda") for x in batch_sentence]
    maxlen = int(max(lengths))
    # depth->[D]; distance->[D,D]
    label_max_shape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_max_shape, device="cuda") for x in word_pos]
    for i, x in enumerate(batch_sentence):
        length = x[1].shape[0]
        if len(label_shape) == 1:
            labels[i][:length] = x[1]
        elif len(label_shape) == 2:
            labels[i][:length, :length] = x[1]
        else:
            raise ValueError(
                "Labels must be either 1D or 2D right now; got either 0D or >3D"
            )
    labels = torch.stack(labels)

    # sentences --> model --> embedding
    # words用于组合词向量
    # labels --> loss --> backward
    # lengths用于packing
    return sentences, words, word_pos, word_len, lengths, labels
