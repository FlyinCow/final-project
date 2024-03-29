# from transformers import AutoTokenizer, AutoModel

# model = AutoModel.from_pretrained("bert-base-chinese")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# input = tokenizer("今天天气真好。", return_tensors="pt").to("cuda")
# model.to("cuda")
# print(model(**input).last_hidden_state.shape)


from collections import namedtuple
from tqdm import tqdm


def generate_lines_for_sent(lines):
    buf = []
    for i, line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append((i, line.strip()))
    if buf:
        yield buf


observation_class = namedtuple(
    "Fields",
    [
        "index",
        "sentence",
        "lemma_sentence",
        "upos_sentence",
        "xpos_sentence",
        "morph",
        "head_indices",
        "governance_relations",
        "secondary_relations",
        "extra_info",
    ],
)

lines = [
    (i + 1, x)
    for i, x in tqdm(
        enumerate(open("data/SemEval-2016/test/text.test.conll")),
        desc="[reading lines]",
    )
]
# res = []
errlines = []
for buf in generate_lines_for_sent(lines):
    conllx_lines = []
    for line in buf:
        conllx_lines.append((line[0], line[1].strip().split("\t")))
    last = 0
    for line in conllx_lines:
        if int(line[1][0]) == last:
            errlines.append(line[0])
        else:
            last = int(line[1][0])
print("[lines read count]:{}".format(len(lines)))
print("[errlines count]:{}".format(len(errlines)))

lines_wirted_count = 0
with open(
    "data/SemEval-2016/test/text.test.deduplicated.conll", "w+", encoding="utf-8"
) as fw:
    for i, line in tqdm(lines, desc="[writing lines]"):
        if i not in errlines:
            # print(i, line)
            fw.write(line)
            lines_wirted_count += 1

print("[lines writed count]:{}".format(lines_wirted_count))
#     sentence = [*zip(*conllx_lines)]
#     res.append(observation_class(*sentence))

# for sent in res[0:3]:
#     print("".join(sent.sentence))
