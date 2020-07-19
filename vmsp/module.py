import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'t h i s </w>':1, 'i s </w>':2, 's w e e t </w>':2, 'p o t a t o </w>':1, 'i t </w>':1}

vocab = {'h e </w>': 1, 'l o v e s </w>': 2, 'h e r </w>': 1, 's h e </w>': 1, 'h i m </w>': 1}

vocab = {'h e l l o':1, 'i':2, 'a m':1, 'h e e j e o n g':1, 'l i k e':1, 'j e l l y': 1}
num_merges = 3
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)






def getTestHDB(n_item_types, n_records, max_seq_len, max_n_items):
    from numpy.random import choice, randint, seed
    print("테스트 HDB를 생성합니다.")
    print("  아이템 종류 개수: %d" % n_item_types)
    print("  레코드 수: %d" % n_records)
    print("  레코드 한개의 시퀀스 최대 길이: %d" % max_seq_len)
    print("  시퀀스 한개에서 한 시점의 최대 아이템 개수: %d" % max_n_items)
    print()
    seed(0)
    items = list('abcdefghijklmnopqrstuvwxyz'[:n_item_types])[::-1]
    hdb = [[list(choice(items, randint(max_n_items)+1, replace=False))
            for _ in range(randint(max_seq_len)+1)]
           for _ in range(n_records)]
    return hdb


# A: 아이템 종류 개수
# B: 데이터베이스 레코드 개수
# C: 시퀀스의 최대 요소 개수
class ZeroGapVMSP():

    def __init__(self):
        print("연속하는 패턴만 고려합니다.\n")

    def runAlgorithm(self, hdb, minsup):
        import time
        print("Construct data structures. (1/3)")
        s = time.time()
        self._initialize(hdb, minsup)
        print('%f (sec)\n' % (time.time()-s))
        print("Build CMAP. (2/3)")
        s = time.time()
        self._setCMAP()
        print('%f (sec)\n' % (time.time()-s))
        print("Extract maximal sequences. (3/3)")
        s = time.time()
        self._searchMaximalSeq()
        print('%f (sec)\n' % (time.time()-s))

    # Big-O: B*C
    def _initialize(self, hdb, minsup):
        minsup = max([1, int(len(hdb)*minsup)])
        items = sorted(set([v__ for v in hdb for v_ in v for v__ in v_]))
        self.item2idx = {v: i for i, v in enumerate(items)}
        vdb = {((i,),): {i_: [] for i_ in range(len(hdb))} for i in items}
        self.items = items
        for idx, seq in enumerate(hdb):
            for idx_, items_ in enumerate(seq):
                for item_ in items_:
                    vdb[((item_,),)][idx].append(idx_)
        self.hdb = hdb
        seq2info = {k: self._getInfo(k, v) for k, v in vdb.items()}
        for seq in list(seq2info):
            if seq2info[seq]['sup'] < minsup:
                del vdb[seq], seq2info[seq]
        self.minsup = minsup
        self.seq2info = seq2info
        freq_items = [v[0][0] for v in vdb]
        self.zs = [sorted(vdb, key=lambda k: seq2info[k]['sum'])]
        self._initial_vdb = vdb
        self._freq_items = freq_items
        self._max_seq_len = max(sum(1 for v_ in v for v__ in v_) for v in hdb)

    # Big-O: B*C*A**2
    def _setCMAP(self):
        vdb, freq_items = self._initial_vdb, self._freq_items
        getInfo = self._getInfo
        minsup, item2idx = self.minsup, self.item2idx
        scmap = {k: [] for k in freq_items}
        icmap = {k: [] for k in freq_items}
        for seq in vdb:
            key = seq[-1][-1]
            for item in freq_items:
                info = getInfo(*self._getSExtension(vdb, seq, item))
                if info['sup'] >= minsup:
                    scmap[key].append(item)
            for item in [v for v in freq_items if item2idx[key] < item2idx[v]]:
                info = getInfo(*self._getIExtension(vdb, seq, item))
                if info['sup'] >= minsup:
                    icmap[key].append(item)
        self.scmap, self.icmap = scmap, icmap

    # Big-O: B*C**2*{Pruned A}**C
    def _searchMaximalSeq(self):
        import sys

        def print_log(n_seqs, zs_len, iter_):
            max_seq_len = self._max_seq_len
            log = (zs_len-1)/max_seq_len+1/max_seq_len*iter_/n_seqs
            sys.stdout.write('\r  %0.8f/1 (진행중)' % log)

        zs, seq2info = self.zs, self.seq2info
        prev_vdb = self._initial_vdb.copy()
        s_opt = {'func': self._getSExtension, 'cmap': self.scmap}
        i_opt = {'func': self._getIExtension, 'cmap': self.icmap}
        print("  다음 시퀀스가 없을경우 조기 중단합니다.")
        while prev_vdb:
            iter_ = 0
            sub_seqs, vdb = [], {}
            for prev_seq in prev_vdb:
                for opt in [s_opt, i_opt]:
                    for item in opt['cmap'][prev_seq[-1][-1]]:
                        seq, sid2idxs = opt['func'](prev_vdb, prev_seq, item)
                        info = self._getInfo(seq, sid2idxs)
                        if info['sup'] >= self.minsup:
                            seq2info[seq], vdb[seq] = info, sid2idxs
                if vdb:
                    sub_seqs.append(prev_seq)
                iter_ += 1
                print_log(len(prev_vdb), len(zs), iter_)
            for sub_seq in sub_seqs:
                del prev_vdb[sub_seq], seq2info[sub_seq]
            zs[-1] = sorted(prev_vdb, key=lambda k: seq2info[k]['sum'])
            zs.append(sorted(vdb, key=lambda k: seq2info[k]['sum']))
            for seq in vdb:
                for idx in range(len(zs)-2):
                    z, new_z = zs[idx], []
                    for idx_, prev_seq in enumerate(z):
                        info, info_ = seq2info[seq], seq2info[prev_seq]
                        if info['sum'] <= info_['sum']:
                            new_z += z[idx_:]
                            break
                        if info['sup'] > info_['sup']:
                            new_z.append(prev_seq)
                        elif self._isSub(seq, prev_seq):
                            del seq2info[prev_seq]
                        else:
                            new_z.append(prev_seq)
                    zs[idx] = new_z
            print()
            print('    시퀀스 크기 \'%d\'에서 발견된 개수: %d' % (len(zs), len(vdb)))
            print('    현재까지 취합된 개수: %d' % len(seq2info))
            prev_vdb = vdb
        self.maximal_seqs = [v_ for v in zs for v_ in v]

    # Big-O: {B or C}
    def _getInfo(self, seq, val):
        n_items = sum(1 for v in seq for v_ in v)
        items_sum = sum(self.item2idx[v_] for v in seq for v_ in v)
        sup = sum(1 for k in val if val[k])
        return {'nItems': n_items, 'sum': items_sum, 'sup': sup}

    # Big-O: B*C
    def _getSExtension(self, vdb, seq, item):
        sid2idxs = {k: [] for k in vdb[seq]}
        case2dic = {'L': vdb[seq], 'S': self._initial_vdb[((item,),)]}
        new_seq = tuple(list(seq)+[(item,)])
        for sid in sid2idxs:
            idxs, idxs_ = case2dic['L'][sid], case2dic['S'][sid]
            loc = loc_ = 0
            while loc < len(idxs) and loc_ < len(idxs_):
                idx, idx_ = idxs[loc], idxs_[loc_]
                if idx+1 == idx_:
                    sid2idxs[sid].append(idx_)
                    loc, loc_ = loc+1, loc_+1
                else:
                    if idx+1 < idx_:
                        loc += 1
                    else:
                        loc_ += 1
        return new_seq, sid2idxs

    # Big-O: B*C
    def _getIExtension(self, vdb, seq, item):
        sid2idxs = {k: [] for k in vdb[seq]}
        case2dic = {'L': vdb[seq], 'S': self._initial_vdb[((item,),)]}
        seq, items = list(seq), tuple(list(seq[-1])+[item])
        seq.pop()
        seq.append(items)
        new_seq = tuple(seq)
        for sid in sid2idxs:
            idxs, idxs_ = case2dic['L'][sid], case2dic['S'][sid]
            loc = loc_ = 0
            while loc < len(idxs) and loc_ < len(idxs_):
                idx, idx_ = idxs[loc], idxs_[loc_]
                if idx == idx_:
                    sid2idxs[sid].append(idx_)
                    loc, loc_ = loc+1, loc_+1
                else:
                    if idx < idx_:
                        loc += 1
                    else:
                        loc_ += 1
        return new_seq, sid2idxs

    # Big-O: C
    def _isSub(self, seq, seq_):
        len_ = len(seq_)
        diff = len(seq)-len_
        if diff == 0:
            if not (
                {(i, v_) for i, v in enumerate(seq_) for v_ in v} -
                {(i, v_) for i, v in enumerate(seq) for v_ in v}
            ):
                return True
        else:
            for idx in range(diff+1):
                if seq[idx:idx+len_] == seq_:
                    return True
        return False
