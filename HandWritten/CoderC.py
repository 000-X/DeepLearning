class CoderC:
    """
    定义两个方法， 用于字符编码和解码。
    """

    def __init__(self):
        with open(r'H:\pro_yzy\HandWritten\Generate\txt\Characters.txt', 'r', encoding='utf-8') as file:
            self.char_set = [line.strip() for line in file if line.strip() != '']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set, start=2)}
        self.char_to_idx['NULL'] = 0
        self.char_to_idx[None] = 1
        self.idx_to_char = {idx: char for idx, char in enumerate(self.char_set)}

    def get_set_len(self):
        return len(self.char_set)

    def get_idx(self, char):
        if char == 'NULL':
            return self.char_to_idx['NULL']
        return self.char_to_idx.get(char, self.char_to_idx[None])

    def get_char(self, idx):
        return self.idx_to_char.get(idx)
