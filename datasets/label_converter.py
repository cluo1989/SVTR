'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-24 19:03:41
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-25 14:39:46
FilePath: /SVTR/datasets/label_converter.py
Description: 
'''
from datasets.charset import alphabet as charset
from datasets.error_label_dict import errlabel

word2indexs = {}
indexs2word = {}

for idx, char in enumerate(charset):
    word2indexs[char] = idx
    indexs2word[idx] = char

def encode(word):
    indexs = []
    for c in word:
        try:
            if c in errlabel:
                c = errlabel[c]

            if len(c) == 1:
                # idx = charset.index(c)
                idx = word2indexs[c]
                indexs.append(idx)
            else:
                for cc in c:
                    # idx = charset.index(cc)
                    idx = word2indexs[cc]
                    indexs.append(idx)
        except:
            # raise ValueError(f'Char Encode Failed: {c}')
            return None

    return indexs

def decode(indexs):
    chars = []
    for idx in indexs:
        chars.append(charset[idx])

    return ''.join(chars)


if __name__ == '__main__':
    word = 'Ⅱ PyTorch 多机多卡训练方法。'#時'
    indexs = encode(word)
    print(indexs)

    word = decode(indexs)
    print(word)
