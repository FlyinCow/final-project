from model.dataset import ConllDataset
from model.tasks import DepthTask, DistanceTask

prefix = "data/SemEval-2016"

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

tasks = {
    "distance": DistanceTask,
    "depth": DepthTask,
}

for genre in ["news", "text"]:
    print('processing dataset ..."{}"'.format(genre))
    for split in ["train", "validation", "test"]:
        print("processing {} set ...".format(split))
        dataset = ConllDataset(dataset_path[genre][split])
        print("[depth task labels]")
        dataset.set_labels(DepthTask())
        p = prefix + "/{split}/{genre}.depth.pt".format(split=split, genre=genre)
        print("saving to {}".format(p))
        dataset.save_labels(p)
        print("[distance task labels]")
        dataset.set_labels(DistanceTask())
        p = prefix + "/{split}/{genre}.distance.pt".format(split=split, genre=genre)
        dataset.save_labels(p)
