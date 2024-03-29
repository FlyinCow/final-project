from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from model.models import PretrainedModel
from model.dataset import ConllDataset, padding_collate_fn
from model.tasks import DistanceTask, DepthTask
from model.probe import DistanceProbe, DepthProbe
from model.loss import L1DistanceLoss, L1DepthLoss

dataset_path = {
    "news": {
        "train": "data/SemEval-2016/train/news.train.deduplicated.conll",
        "validation": "data/SemEval-2016/validation/news.valid.deduplicated.conll",
        "test": "data/SemEval-2016/test/news.test.deduplicated.conll",
    },
    "text": {
        "train": "data/SemEval-2016/train/text.train.deduplicated.conll",
        "validation": "data/SemEval-2016/validation/text.valid.deduplicated.conll",
        "test": "data/SemEval-2016/test/text.test.deduplicated.conll",
    },
}

dataset = ConllDataset(dataset_path["news"]["test"])
dataset.set_labels(DistanceTask())

dataloader = DataLoader(dataset, batch_size=5, collate_fn=padding_collate_fn)

bert_base_chinese_model = PretrainedModel("bert-base-chinese")

probe = DistanceProbe(300, 768)

loss_function = L1DistanceLoss()
optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)

# training loop
for sentences, words, word_pos, word_len, lengths, labels in tqdm(
    dataloader, desc="[training batch]"
):
    optimizer.zero_grad()
    # 1. 从待探测的模型获取词向量
    embs = bert_base_chinese_model(sentences, word_pos, word_len, lengths)
    # 2. 使用探针获取距离/深度
    predictions = probe(embs)
    # 3. 计算Loss, 反向传播
    batch_loss = loss_function(predictions, labels, lengths)
    batch_loss.backward()
    optimizer.step()
