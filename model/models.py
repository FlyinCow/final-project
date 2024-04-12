"""Classes for constructing word representations."""

# from pytorch_pretrained_bert import BertModel
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
from functools import partial


class ProjectionModel(nn.Module):
    """A class for simple contextualization of word-level embeddings.
    Runs an untrained BiLSTM on top of the loaded-from-disk embeddings.
    """

    def __init__(self, args):
        super(ProjectionModel, self).__init__(args)
        input_dim = args["model"]["hidden_dim"]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=int(input_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        for param in self.lstm.parameters():
            param.requires_grad = False
        self.lstm.to(args["device"])

    def forward(self, batch):
        """Random BiLSTM contextualization of embeddings

        Args:
          batch: a batch of pre-computed embeddings loaded from disk.

        Returns:
          A random-init BiLSTM contextualization of the embeddings
        """
        with torch.no_grad():
            projected, _ = self.lstm(batch)
        return projected


class DecayModel(nn.Module):
    """A class for simple contextualization of word-level embeddings.
    Computes a weighted average of the entire sentence at each word.

    """

    def __init__(self, args):
        super(DecayModel, self).__init__(args)
        self.args = args

    def forward(self, batch):
        """Exponential-decay contextualization of word embeddings.

        Args:
          batch: a batch of pre-computed embeddings loaded from disk.

        Returns:
          An exponentially-decaying average of the entire sequence as
          a representation for each word.
          Specifically, for word i, assigns weight:
            1 to word i
            1/2 to word (i-1,i+1)
            2/4 to word (i-2,i+2)
            ...
          before normalization by the total weight.
        """
        forward_aggregate = torch.zeros(*batch.size(), device=self.args["device"])
        backward_aggregate = torch.zeros(*batch.size(), device=self.args["device"])
        forward_normalization_tensor = torch.zeros(
            batch.size()[1], device=self.args["device"]
        )
        backward_normalization_tensor = torch.zeros(
            batch.size()[1], device=self.args["device"]
        )
        batch_seq_len = torch.tensor(batch.size()[1], device=self.args["device"])
        decay_constant = torch.tensor(0.5, device=self.args["device"])
        for i in range(batch_seq_len):
            if i == 0:
                forward_aggregate[:, i, :] = batch[:, i, :]
                backward_aggregate[:, batch_seq_len - i - 1, :] = batch[
                    :, batch_seq_len - i - 1, :
                ]
                forward_normalization_tensor[i] = 1
                backward_normalization_tensor[batch_seq_len - i - 1] = 1
            else:
                forward_aggregate[:, i, :] = (
                    forward_aggregate[:, i - 1, :] * decay_constant
                ) + batch[:, i, :]
                backward_aggregate[:, batch_seq_len - i - 1, :] = (
                    backward_aggregate[:, batch_seq_len - i, :] * decay_constant
                ) + batch[:, batch_seq_len - i - 1, :]
                forward_normalization_tensor[i] = (
                    forward_normalization_tensor[i - 1] * decay_constant + 1
                )
                backward_normalization_tensor[batch_seq_len - i - 1] = (
                    backward_normalization_tensor[batch_seq_len - i] * decay_constant
                    + 1
                )
        normalization = forward_normalization_tensor + backward_normalization_tensor
        normalization = normalization.unsqueeze(1).unsqueeze(0)
        decay_aggregate = (forward_aggregate + backward_aggregate) / normalization
        return decay_aggregate


class TransformersPretrainedModel(nn.Module):
    """使用预训练语言模型从原始句子生成字向量，然后将词的字向量取平均作为词向量"""

    def __init__(self, model_used="bert-base-chinese"):
        super(TransformersPretrainedModel, self).__init__()
        # todo: 测试除了`bert-base-chinese`之外的其它模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_used)
        self.pretrained_model = AutoModel.from_pretrained(model_used)
        self.pretrained_model.to("cuda")
        self.pretrained_model.eval()

    def forward(self, sentences, word_pos, word_len, lengths):
        # todo: 处理[unk]和英文字符
        # model_input = self.tokenizer(sentences, return_tensors="pt", padding=True).to(
        #     "cuda"
        # )
        model_input = {"input_ids": []}
        for i, sent in enumerate(sentences):
            model_input["input_ids"].append(
                self.tokenizer.convert_tokens_to_ids([c for c in sent])
            )
            if self.tokenizer.unk_token_id in model_input["input_ids"]:
                lengths[i] = 0
        # todo: check有没有用
        with torch.no_grad():
            model_output = self.pretrained_model(**model_input)
            # 取到hidden_states，组合出词的向量
            output = []
            # todo: 获取其他层的词向量
            char_embs = model_output.last_hidden_state

            for i, char_embs_i in enumerate(char_embs):
                word_len_i = word_len[i][: lengths[i]]
                word_pos_i = word_pos[i][: lengths[i]]
                grouped = torch.split(
                    char_embs_i[: len(sentences[i])],
                    word_len_i.tolist(),
                )
                # torch.stack(list(map(partial(torch.sum, dim=0), grouped))) / word_len_i
                output.append(
                    torch.stack(list(map(partial(torch.sum, dim=0), grouped)))
                    / torch.unsqueeze(word_len_i, dim=1)
                )

                # output.append(
                #     torch.tensor(
                #         list(map(partial(torch.sum, dim=0), grouped)), device="cuda"
                #     )
                #     / word_len[i][: lengths[i]]
                # )
            output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
        return output
