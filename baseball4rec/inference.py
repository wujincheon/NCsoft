import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from dataloader import DataSets
from train import evaluate
from model import Baseball4Rec

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import pickle
import utils
import logging
import argparse
import random
import os
import warnings
import itertools

from random import *
from PIL import Image
 

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

# 어텐션 스코어에서 각 변수에 대해 원래의 피처이름으로 반환하기 위한 딕셔너리
idx2name={0: 'is_foreigner',
 1: 'pit_hand_side',
 2: 'pit_hand_type',
 3: 'FAST_ratio',
 4: 'CURV_ratio',
 5: 'FORK_ratio',
 6: 'SLID_ratio',
 7: 'TWOS_ratio',
 8: 'CHUP_ratio',
 9: 'CUTT_ratio',
 10: 'SINK_ratio',
 11: 'KNUK_ratio',
 12: 'is_foreigner',
 13: 'bat_hand',
 14: 'FAST_Hitrate',
 15: 'CURV_Hitrate',
 16: 'FORK_Hitrate',
 17: 'SLID_Hitrate',
 18: 'TWOS_Hitrate',
 19: 'CHUP_Hitrate',
 20: 'CUTT_Hitrate',
 21: 'SINK_Hitrate',
 22: 'KNUK_Hitrate',
 23: 'inn',
 24: 'rstate',
 25: 'bat_order',
 26: 'out_snt',
 27: 'strike_cnt',
 28: 'ball_cnt',
 29: 'pit_total_cnt',
 30: 'pit_tic_cnt',
 31: 'pit_score',
 32: 'bat_score'}

# confusion map을 그리는 함수
def draw_cm(cm, save_dir, balls):
    fig, ax = plt.subplots(figsize=(6, 5))
    fontprop = fm.FontProperties(size=15)

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    balls = balls
    tick_marks = np.arange(len(balls))

    ax = plt.gca()
    plt.xticks(tick_marks)
    ax.set_xticklabels(balls, fontproperties=fontprop)
    plt.yticks(tick_marks)
    ax.set_yticklabels(balls, fontproperties=fontprop)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )
    plt.tight_layout()
    plt.ylabel("True", fontsize=12)
    plt.xlabel("Predict", fontsize=12)

    plt_dir = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(plt_dir, dpi=300, bbox_inches="tight")

    logger.info("  Confusion Matrix are saved to, %s", plt_dir)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="../data/rec_binary_data_loc_test.pkl", type=str, help="test data path"
    )
    parser.add_argument("--model_path", default="../output_loc/pytorch_model.bin", type=str, help="model path")
    parser.add_argument("--args_path", default="../output_loc/training_args.bin", type=str, help="args path")
    k= randint(0, 28335)  # test data 중 한 개를 골라서 그려보기 위한 변수

    args_ = parser.parse_args()
    args = torch.load(args_.args_path)
    args.test_data_path = args_.data_path
    args.eval_batch_size = 256

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # 앞서 구했듯이, 트레이닝 데이터에 존재하는 타자와 투수의 unique한 수와, 그를 index로 바꾸기 위한 리스트를 전달
    with open('../data/id_list.csv', newline='') as f:
        reader = csv.reader(f)
        id_list = list(reader)
    batter_len=len(set(id_list[0]))
    pitcher_len=len(set(id_list[1]))
    
    dataset = DataSets(args.test_data_path,id_list)
    # subtract two index variables and three categorical variables
    n_pitcher_cont = 9
    # subtract two index variables and two categorical variables
    n_batter_cont = 9
    # subtract one index variables and two categorical variables
    n_state_cont = len(dataset.state[0]) - 3

    model = Baseball4Rec(
        n_pitcher_cont,
        pitcher_len,        
        n_batter_cont,
        batter_len,
        n_state_cont,
        
        n_encoder_layer=args.n_encoder_layer,
        n_decoder_layer=args.n_decoder_layer,
        n_concat_layer=args.n_concat_layer,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)
    
    
    model.load_state_dict(torch.load(args_.model_path), strict=False)

    model.eval()
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.eval_batch_size,)
    result, f1_log_3, f1_log_9, cm, cm2, top3, top3_score, att_w, att_w2, recall5, mrr5, recall10, mrr10, recall20, mrr20 = evaluate(args, model, id_list, k) # top3 : [3], att_w : [155] , att_w2 : [3]
    #인덱스 다시 구질로
    idx2ball = {j:i for i,j in ball2idx.items()}
    

   
    #print("attention score")
    #print(att_w)
    #print(att_w2)
    
    ### 테스트 데이터에 대해서 임의의 k번째 데이터의 결과를 예측하고, 그에 대한 예측 스코어 및 어텐션 스코어를 시각화하여 이미지로 저장하는 과정
    
    ## 1) figure 및 axis 생성하기
    fig, ax = plt.subplots(2,1,figsize=(5,5),gridspec_kw={'height_ratios':[2,1]})
    
    ## 2) image 출력하기
    img=Image.open("label.png")
    ax[0].imshow(img)
    ax[0].axis('off')
    title= '1. '+idx2ball[top3.tolist()[0]]+ ' (%f)'%top3_score.tolist()[0] +'\n2. ' + idx2ball[top3.tolist()[1]]+ ' (%f)'%top3_score.tolist()[1] +'\n3. ' + idx2ball[top3.tolist()[2]] + ' (%f)'%top3_score.tolist()[2]
    ax[0].set_title(title, loc='center')
    
    
    
    ## 3) att score
    x= np.array(att_w.tolist())
    att_top5 = torch.topk(att_w.detach().cpu(), k=5, dim=0)[1]
    print(att_top5.tolist())

    ax[1].imshow(x[np.newaxis,:], cmap="Blues")#, extent=extent , aspect="auto"
    
    
    
    
    ax[1].set_yticks([])
    title2= idx2name[att_top5.tolist()[0]] +', ' + idx2name[att_top5.tolist()[1]] +', ' + idx2name[att_top5.tolist()[2]] +', ' + idx2name[att_top5.tolist()[3]] +', ' + idx2name[att_top5.tolist()[4]]
    ax[1].set_title(title2, loc='center',fontsize=10)
    
    
    
    
    ## 4) 여백 다듬기
    plt.tight_layout()
    
    ## 5) 그래프 저장하기
    plt.savefig("result_visual")
    
    ## 6) 히트맵 따로
    plt.figure(figsize=(30,30))
    y= x.reshape(1,33)
    df=pd.DataFrame(y, columns=list(idx2name.values()), index=[''])
    
    
    plt.imshow(df, cmap='YlGnBu')
    plt.colorbar(fraction=0.012, pad=0.04)
    plt.xticks(range(33),df.columns, rotation=20)
    plt.yticks(range(1),df.index)
    
    plt.savefig("result_visual_att")
    
    
    
    
    print('Recall@{}: {:.4f}, MRR@{}: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f}  \n'.format(5, recall5, 5, mrr5, 10, recall10, 10, mrr10, 20, recall20, 20, mrr20))
        
    print(
        
        "loss : {}  ".format(
            result["loss"]),
        "f1-macro-3 : {}  ".format(
                    result["pitch_macro_f1_3"]),
        "f1-micro-3 : {}  ".format(
                    result["pitch_micro_f1_3"]),

        
        "f1-macro-9 : {}  ".format(
                    result["pitch_macro_f1_9"]),
        "f1-micro-9 : {}  ".format(
                    result["pitch_micro_f1_9"],

        )
    )
    result_dir = os.path.join("test_results.txt")
    utils.print_result(result_dir, result, f1_log_3, f1_log_9)
    balls3=['Fast', 'Horizon', 'Vertical']
    balls9=['CHUP', 'CURV', 'CUTT', 'FAST', 'FORK', 'KNUC', 'SINK', 'SLID', 'TWOS']
    draw_cm(cm,'cm_9',balls9)
    draw_cm(cm2,'cm_3',balls3)
    # tmp = pd.read_csv(args.test_data_path)
    # tmp["Pred"] = [label_list[i] for i in total_y_hat]
    # tmp.to_csv("./result/test_result.csv")
    # print("results are saved to result folder")


if __name__ == "__main__":
    main()
