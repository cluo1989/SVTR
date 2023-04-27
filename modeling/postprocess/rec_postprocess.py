'''
Author: Cristiano-3 chunanluo@126.com
Date: 2023-04-26 12:41:40
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2023-04-26 14:26:01
FilePath: /SVTR/modeling/postprocess/rec_postprocess.py
Description: 
'''
from datasets.charset import alphabet
BLANK_INDEX = len(alphabet)


class BaseRecLabelDecode(object):
    def __init__(self) -> None:
        pass

    def decode(self, pred, probs):
        raise NotImplementedError


class CTCDecode(BaseRecLabelDecode):
    def __init__(self):
        super().__init__()

    def decode(self, pred, probs):
        argmax_idx = pred  # np.argmax(pred, axis=1)

        res_dict=[]
        last_is_cut = False
        for idx, char_idx in enumerate(argmax_idx):
            if char_idx != BLANK_INDEX:
                # get max prob and it's char
                prob = probs[idx]#pred[idx][char_idx]
                char = alphabet[char_idx]
                item = [char, prob, 1]

                # first char, 
                # append a new char
                if len(res_dict) == 0:
                    res_dict.append(item)
                    last_is_cut = False
                    continue

                # if last is cut, 
                # append a new char
                if last_is_cut:
                    res_dict.append(item)
                # if different with last char, 
                # append a new char
                elif char != res_dict[-1][0]:
                    res_dict.append(item)
                # if equal to last char,
                # accumulate prob and count
                else:
                    res_dict[-1][1] += prob
                    res_dict[-1][2] += 1

                last_is_cut = False

            else:
                last_is_cut = True

        res_dict = [[itm[0], itm[1]/itm[2]] for itm in res_dict]
        results_str = ''.join([t[0] for t in res_dict])
        text_probs = [t[1] for t in res_dict]
        return results_str
        