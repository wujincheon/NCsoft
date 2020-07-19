from pprint import pprint

from module import ZeroGapVMSP, getTestHDB


if __name__ == '__main__':

    paper_hdb = [
        [['a', 'b'], ['c'], ['f', 'g'], ['g'], ['e']],
        [['a', 'd'], ['c'], ['b'], ['a', 'b', 'e', 'f']],
        [['a'], ['b'], ['f', 'g'], ['e']],
        [['b'], ['f', 'g']]
    ]
    proc = ZeroGapVMSP()
    proc.runAlgorithm(paper_hdb, 0.5)
    pprint(proc.maximal_seqs)
    
    print()
    print("===================================================================")
    print()
    
    hdb = getTestHDB(10, 1000, 5, 4)
    proc = ZeroGapVMSP()
    proc.runAlgorithm(hdb, 0.01)

    len_ = len(proc.maximal_seqs)
    for i in range(4):
        idx = len_//4*(i)
        print(idx, proc.maximal_seqs[idx])
    print(len_, proc.maximal_seqs[-1])
