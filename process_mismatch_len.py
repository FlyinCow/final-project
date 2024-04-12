from model.dataset import ConllDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from model.tasks import EmptyTask

# s = "三峡工程管理系统将引进加拿大ＭＡＩ公司科学的管理办法和先进的计算机技术，以工程数据库管理系统为核心，对三峡工程的各分项工程的设计、计划、合同、财务、物资、设备、施工、安装的全过程进行控制和管理。"


def test_unk(genre: str, split: str):
    print("[testing {}/{}]".format(genre, split))

    dataset_path = {
        "news": {
            "train": "data/SemEval-2016/train/news.train.deduplicated.nounk.conll",
            "validation": "data/SemEval-2016/validation/news.valid.deduplicated.nounk.conll",
            "test": "data/SemEval-2016/test/news.test.deduplicated.nounk.conll",
        },
        "text": {
            "train": "data/SemEval-2016/train/text.train.deduplicated.nounk.conll",
            "validation": "data/SemEval-2016/validation/text.valid.deduplicated.nounk.conll",
            "test": "data/SemEval-2016/test/text.test.deduplicated.nounk.conll",
        },
    }
    datapath = dataset_path[genre][split]

    dataset = ConllDataset(datapath)
    # dataset = ConllDataset("data/SemEval-2016/train/news.train.deduplicated.nounk.conll")
    dataset.set_labels(EmptyTask())

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # tokenizer = AutoTokenizer.from_pretrained("uer/chinese_roberta_L-8_H-512")
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")

    unk_count = 0
    unks = []
    todelete = [False for _ in range(dataset.current_line_no)]
    for i, s in tqdm(enumerate(dataset), desc="[collecting sentences]"):
        raw_sentence = s[0]["raw_sentence"]
        model_input = tokenizer.convert_tokens_to_ids([c for c in raw_sentence])
        if tokenizer.unk_token_id in model_input:
            unks.append(
                (dataset.start_lines[i] - s[0]["words_count"], dataset.start_lines[i])
            )
            for ii in range(
                dataset.start_lines[i] - s[0]["words_count"], dataset.start_lines[i]
            ):
                todelete[ii - 1] = True
            unk_count += 1

    print("mismatch by [unk]: {}/{}".format(unk_count, len(dataset)))


for genre in ["text", "news"]:
    for split in ["train", "test", "validation"]:
        test_unk(genre, split)

# print(unks)

# with open(datapath, "r", encoding="utf-8") as origin_file:
#     name = datapath.split(".")
#     name.insert(-1, "nounk")
#     name = ".".join(name)
#     with open(name, "x", encoding="utf-8") as file_to_write:
#         for i, line in tqdm(enumerate(origin_file), desc="[writing lines]"):
#             file_to_write.write("\n" if todelete[i] else line)
