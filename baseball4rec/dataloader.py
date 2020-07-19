import os
import logging
import pickle
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np

from collections import OrderedDict

logger = logging.getLogger(__name__)
# ball type을 index로 매칭시키는 딕셔너리
ball2idx = {
    "CHUP_0" : 0,
    "CHUP_1" : 1,
    "CHUP_2" : 2,
    "CHUP_3" : 3,
    "CHUP_4" : 4,
    "CHUP_5" : 5,
    "CHUP_6" : 6,
    "CHUP_7" : 7,
    "CHUP_8" : 8,
    "CHUP_9" : 9,
    "CHUP_10" : 10,
    "CHUP_11" : 11,
    "CHUP_12" : 12,
    "CHUP_13" : 13,
    "CHUP_14" : 14,
    "CHUP_15" : 15,
    "CURV_0" : 16,
    "CURV_1" : 17,
    "CURV_2" : 18,
    "CURV_3" : 19,
    "CURV_4" : 20,
    "CURV_5" : 21,
    "CURV_6" : 22,
    "CURV_7" : 23,
    "CURV_8" : 24,
    "CURV_9" : 25,
    "CURV_10" : 26,
    "CURV_11" : 27,
    "CURV_12" : 28,
    "CURV_13" : 29,
    "CURV_14" : 30,
    "CURV_15" : 31,
    "CUTT_0" : 32,
    "CUTT_1" : 33,
    "CUTT_2" : 34,
    "CUTT_3" : 35,
    "CUTT_4" : 36,
    "CUTT_5" : 37,
    "CUTT_6" : 38,
    "CUTT_7" : 39,
    "CUTT_8" : 40,
    "CUTT_9" : 41,
    "CUTT_10" : 42,
    "CUTT_11" : 43,
    "CUTT_12" : 44,
    "CUTT_13" : 45,
    "CUTT_14" : 46,
    "CUTT_15" : 47,
    "FAST_0" : 48,
    "FAST_1" : 49,
    "FAST_2" : 50,
    "FAST_3" : 51,
    "FAST_4" : 52,
    "FAST_5" : 53,
    "FAST_6" : 54,
    "FAST_7" : 55,
    "FAST_8" : 56,
    "FAST_9" : 57,
    "FAST_10" : 58,
    "FAST_11" : 59,
    "FAST_12" : 60,
    "FAST_13" : 61,
    "FAST_14" : 62,
    "FAST_15" : 63,
    "FORK_0" : 64,
    "FORK_1" : 65,
    "FORK_2" : 66,
    "FORK_3" : 67,
    "FORK_4" : 68,
    "FORK_5" : 69,
    "FORK_6" : 70,
    "FORK_7" : 71,
    "FORK_8" : 72,
    "FORK_9" : 73,
    "FORK_10" : 74,
    "FORK_11" : 75,
    "FORK_12" : 76,
    "FORK_13" : 77,
    "FORK_14" : 78,
    "FORK_15" : 79,
    "KNUC_0" : 80,
    "KNUC_1" : 81,
    "KNUC_2" : 82,
    "KNUC_3" : 83,
    "KNUC_4" : 84,
    "KNUC_5" : 85,
    "KNUC_6" : 86,
    "KNUC_7" : 87,
    "KNUC_8" : 88,
    "KNUC_9" : 89,
    "KNUC_10" : 90,
    "KNUC_11" : 91,
    "KNUC_12" : 92,
    "KNUC_13" : 93,
    "KNUC_14" : 94,
    "KNUC_15" : 95,
    "SINK_0" : 96,
    "SINK_1" : 97,
    "SINK_2" : 98,
    "SINK_3" : 99,
    "SINK_4" : 100,
    "SINK_5" : 101,
    "SINK_6" : 102,
    "SINK_7" : 103,
    "SINK_8" : 104,
    "SINK_9" : 105,
    "SINK_10" : 106,
    "SINK_11" : 107,
    "SINK_12" : 108,
    "SINK_13" : 109,
    "SINK_14" : 110,
    "SINK_15" : 111,
    "SLID_0" : 112,
    "SLID_1" : 113,
    "SLID_2" : 114,
    "SLID_3" : 115,
    "SLID_4" : 116,
    "SLID_5" : 117,
    "SLID_6" : 118,
    "SLID_7" : 119,
    "SLID_8" : 120,
    "SLID_9" : 121,
    "SLID_10" : 122,
    "SLID_11" : 123,
    "SLID_12" : 124,
    "SLID_13" : 125,
    "SLID_14" : 126,
    "SLID_15" : 127,
    "TWOS_0" : 128,
    "TWOS_1" : 129,
    "TWOS_2" : 130,
    "TWOS_3" : 131,
    "TWOS_4" : 132,
    "TWOS_5" : 133,
    "TWOS_6" : 134,
    "TWOS_7" : 135,
    "TWOS_8" : 136,
    "TWOS_9" : 137,
    "TWOS_10" : 138,
    "TWOS_11" : 139,
    "TWOS_12" : 140,
    "TWOS_13" : 141,
    "TWOS_14" : 142,
    "TWOS_15" : 143,
}
# 투수의 제한 구질을 masking하기 위한 구질별 index 딕셔너리
ball2idx2 = {
    "CHUP": 0,
    "CURV": 1,
    "CUTT": 2,
    "FAST": 3,
    "FORK": 4,
    "KNUC": 5,
    "SINK": 6,
    "SLID": 7,
    "TWOS": 8,
}



def data_preprocess(data: List[Dict]) -> List:
    game_id, data = list(zip(*data.items()))
    #key_order=['Pitcher', 'Batter', 'State', 'Pitch_loc', 'Pitch_set', 'Event_ID', 'Label']
    #order2index={'Pitcher':0, 'Batter':1, 'State':2, 'Pitch':3, 'Pitch_set':4, 'Event_ID':5, 'Label':6}
    #data3=[]
    #for i in data2:
    #    data3.append({order2index[k]:i[k] for k in key_order})
        
               

    # pitcher, batter, state, pitch, pitch_set, event_id, label 구분
    
    data_split = list(zip(*[list(d.values()) for d in data]))
    
    label, event_id = [data_split.pop() for _ in range(2)]
    pitch_loc = data_split.pop(3)

    # pitcher, batter, state, pitch_set
   
    pitch_counts = list(np.unique(pitch_loc, return_counts=True))
    pitch_counts[0] = [ball2idx[t] for t in pitch_counts[0]]
    pitch_counts = dict(sorted(list(zip(*pitch_counts)), key=lambda x: x[0]))

    
    
    
    data_split2 = [[list(d.items()) for d in data] for data in data_split]
    data_split3 = [[list(zip(*d)) for d in data] for data in data_split2]
    (
        (pitcher_name, pitcher),
        (batter_name, batter),
        (state_name, state),
        (pitcher_set_name, pitch_set),
    ) = [list(zip(*data)) for data in data_split3]

    pitcher_name, batter_name, state_name, pitcher_set_name = [
        list(set(names))[0] for names in (pitcher_name, batter_name, state_name, pitcher_set_name)
    ]

    # 해당 투수가 던질 수 없는 구종 mask
    pitch_avail = list(zip(*pitch_set))[1]
    
    ball_list = list(ball2idx.keys())
    ball_list2 = list(ball2idx2.keys())

    def pitch_mask(x):
        x = x.split("_")
        return [ball_list[i][:4] not in x for i in range(144)]
    
    def pitch_mask_feature(x):
        x = x.split("_")
        return [ball_list2[i][:4] not in x for i in range(9)]

    # 던질 수 없는 구종 : 1, 불가능한 구종 : 0
    # 추후 모델에서 softmax 들어가기 전 mask_fill() 함수로  마스킹 수행
    pitch_unavail_mask = list(map(pitch_mask, pitch_avail))
    pitch_unavail_mask_feature = list(map(pitch_mask_feature, pitch_avail))
    

    return (
        game_id,
        label,
        event_id,
        pitch_loc,
        pitch_unavail_mask,
        pitch_unavail_mask_feature,
        pitcher,
        batter,
        state,
        pitch_counts,
        (pitcher_name, batter_name, state_name, pitcher_set_name),
    )


class DataSets(Dataset):
    def __init__(self, file_path, id_list):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, "cached_" + filename)

        # embedding layer를 위한 dictionary
        # TODO: inn : 1~10(1~9이닝 + alpha)
        
            
    
        
        self.batter_id2idx = dict([(j, i) for i, j in enumerate(set(id_list[0]))])
        self.pitcher_id2idx = dict([(j, i) for i, j in enumerate(set(id_list[1]))])
                
        self.inn2idx = dict([(j, i) if j < 10 else (j, 9) for i, j in enumerate(range(1, 13))])
        self.batorder2idx = dict([(j, i) for i, j in enumerate(range(1, 10))])

        if os.path.exists(cached_features_file):
            # preprocess가 진행된 cached 파일을 불러옵니다.
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                (
                    self.pitcher,
                    self.batter,
                    self.state,
                    self.pitch_loc,
                    self.pitch_mask,
                    self.pitch_mask_feature,
                    
                    self.label,
                    self.pitcher_name,
                    self.batter_name,
                    self.state_name,
                    self.pitcher_set_name,
                    self.pitch_counts,
                ) = pickle.load(handle)
        else:
            # 최초 데이터 로드 시 시간이 걸리는 작업을 수행합니다.
            # 두 번째 실행부터는 cached 파일을 이용합니다.
            logger.info("Creating features from dataset file at %s", directory)

            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            (
                game_id,
                self.label,
                event_id,
                pitch_loc,
                self.pitch_mask,
                self.pitch_mask_feature,

                self.pitcher,
                self.batter,
                state,
                self.pitch_counts,
                (self.pitcher_name, self.batter_name, self.state_name, self.pitcher_set_name),
            ) = data_preprocess(data)

            self.pitch_loc = [ball2idx[b] for b in pitch_loc]
            # state는 모두 integer / str 인데 str 변수도 모두 int형태라 int로 통일
            self.state = [[int(i) for i in s] for s in state]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(
                    (
                        self.pitcher,
                        self.batter,
                        self.state,
                        self.pitch_loc,
                        self.pitch_mask,
                        self.pitch_mask_feature,

                        self.label,
                        self.pitcher_name,
                        self.batter_name,
                        self.state_name,
                        self.pitcher_set_name,
                        self.pitch_counts,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def __len__(self):
        return len(self.pitcher)

    def __getitem__(self, item): # 투수id, 투수 관련 피처, 타자id, 타자 관련 피처, state 피처 정보 전달
        
        pitcher_id = torch.tensor( self.pitcher_id2idx[self.pitcher[item][0]], dtype=torch.long)  
        
        pitcher_discrete = torch.tensor( 
            list(self.pitcher[item][2:5]), dtype=torch.long
        ) ## [not i for i in self.pitch_mask_feature[item]] + 

        pitcher_continuous = torch.tensor(self.pitcher[item][64:], dtype=torch.float) # base = 5:  64: = 던진 구종 비율 9개

        batter_id = torch.tensor( self.batter_id2idx[self.batter[item][0]], dtype=torch.long)
        batter_discrete = torch.tensor(self.batter[item][2:4], dtype=torch.long)
        batter_continuous = torch.tensor(self.batter[item][58:], dtype=torch.float) #base = 4: #58:= 구종별 Hitrate 9개


        state_discrete = torch.tensor(
            [self.inn2idx[self.state[item][0]]]
            + [self.state[item][4]]
            + [self.batorder2idx[self.state[item][5]]],
            dtype=torch.long,
        )
        state_continuous = torch.tensor(
            self.state[item][1:4] + self.state[item][6:], dtype=torch.float
        )

        pitch_mask = torch.tensor(self.pitch_mask[item], dtype=torch.bool)
        pitch_loc = torch.tensor(self.pitch_loc[item], dtype=torch.long)

        label = torch.tensor(self.label[item], dtype=torch.long)
        return (
            pitcher_id,
            pitcher_discrete,  # num of unique values : (2, 2, 2)
            pitcher_continuous,

            batter_id,
            batter_discrete,  # num of unique values : (2, 2)
            batter_continuous,

            state_discrete,  # num of unique values : (10, 8, 9)
            state_continuous,

            pitch_mask,
            pitch_loc,
            label,
        )

