import torch


class Task:
    @staticmethod
    def labels(sentences):
        raise NotImplementedError


class DepthTask(Task):
    @staticmethod
    def labels(sentence):
        # {
        #     "raw_sentence": raw_sentence,
        #     "words_count": word_count,
        #     "char_count": char_count,
        #     "root_idx": root_idx,
        #     "raw_words": raw_words,
        #     "word_pos": word_pos,
        #     "word_len": word_pos,
        #     "head_indexs": head_indexs,
        #     "depths": depths,
        # }
        depths = -torch.ones(sentence["words_count"])
        for i in range(sentence["words_count"]):
            DepthTask.update_depth(
                sentence["root_idx"], i, sentence["head_indexs"], depths
            )
        return depths

    @staticmethod
    def update_depth(root_idx, word_id, head_indexs, depths):
        """
        递归更新每个词的树上深度
        """
        if word_id == root_idx:
            # 是root
            depths[word_id] = 0
            return
        if depths[word_id] > 0:
            # 已经更新
            return
        else:
            # 尚未更新
            DepthTask.update_depth(root_idx, head_indexs[word_id], head_indexs, depths)
            depths[word_id] = depths[head_indexs[word_id]] + 1


class DistanceTask(Task):
    # todo: 计算词间距离

    @staticmethod
    def labels(sentences):
        # {
        #     "raw_sentence": raw_sentence,
        #     "words_count": word_count,
        #     "char_count": char_count,
        #     "root_idx": root_idx,
        #     "raw_words": raw_words,
        #     "word_pos": word_pos,
        #     "word_len": word_pos,
        #     "head_indexs": head_indexs,
        #     "depths": depths,
        # }
        words_count = sentences["words_count"]
        distances = -torch.ones([words_count, words_count])
        for i in range(words_count):
            distances[i][i] = 0
        for i in range(words_count):
            for j in range(i):
                DistanceTask.update_distance(
                    i, j, sentences["head_indexs"], sentences["root_idx"], distances
                )
        return distances

    @staticmethod
    def update_distance(i, j, head_indexs, root_idx, distances):
        if i == j:
            return
        if distances[i][j] != -1:
            return
        i_parent = head_indexs[i]
        j_parent = head_indexs[j]
        if i_parent >= 0 and distances[i_parent][j] != -1:
            distances[i][j] = distances[i_parent][j] + 1
            distances[j][i] = distances[i_parent][j] + 1
            return
        if j_parent >= 0 and distances[i][j_parent] != -1:
            distances[i][j] = distances[i][j_parent] + 1
            distances[j][i] = distances[i][j_parent] + 1
            return
        if i != root_idx:
            DistanceTask.update_distance(i_parent, j, head_indexs, root_idx, distances)
            distances[i][j] = distances[i_parent][j] + 1
            distances[j][i] = distances[i_parent][j] + 1
            return
        else:
            DistanceTask.update_distance(i, j_parent, head_indexs, root_idx, distances)
            distances[i][j] = distances[i][j_parent] + 1
            distances[j][i] = distances[i][j_parent] + 1
            return
