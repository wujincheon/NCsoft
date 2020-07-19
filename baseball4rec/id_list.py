import pickle
import csv

# 투수와 타자 id에 대해, index를 0부터 할당하기 위해 모든 투수id와 모든 타자id를 가져옴
with open('../data/rec_binary_data_loc_total_filtered.pkl', 'rb') as f:
    data = pickle.load(f)
    
batter_id=[]
for i in (data.keys()):
    batter_id.append(data[i]['Batter']['batter_id'])
    
pitcher_id=[]
for i in (data.keys()):
    pitcher_id.append(data[i]['Pitcher']['pitcher_id'])
    


with open("../data/id_list.csv", "w",newline='') as f:
    writer = csv.writer(f)
    writer.writerows([batter_id])
    writer.writerows([pitcher_id])
    