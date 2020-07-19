import logging
import argparse
import os
import glob
import csv

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report



from dataloader import DataSets
from model import Baseball4Rec
from utils import set_seed, rotate_checkpoints
import metric


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

try:
    from warmup_scheduler import GradualWarmupScheduler
except ImportError:
    raise ImportError(
        "Please install warmup_scheduler from https://github.com/ildoonet/pytorch-gradual-warmup-lr to use gradual warmup scheduler."
    )

logger = logging.getLogger(__name__)

def train(args, train_dataset, model):
    tb_writer = SummaryWriter()
    counts = train_dataset.pitch_counts
    logger.info("  Counts of each ball type : %s", counts)
    
    #weigted 샘플링 사용시 아래의 sampler를 사용
    weights = [0 if p == 5 or p == 6 else 1.0 / counts[p] for p in train_dataset.pitch_loc]
    #sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    
    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset, sampler=sampler, batch_size=args.train_batch_size,
    )
   

    t_total = len(train_dataloader) * args.num_train_epochs
    args.warmup_step = int(args.warmup_percent * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.Adam(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    if args.warmup_step != 0:
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warmup_step)
    else:
        scheduler = CosineAnnealingLR(optimizer, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    loss_fct = torch.nn.CrossEntropyLoss()

    # Train!
    logger.info("***** Running Baseball Recommendation *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Warmup Steps = %d", args.warmup_step)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch",)
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator): # training 데이터를 배치 단위로 받아서 모델의 input으로 사용, output으로 나온 최종 예측 score를 이용해 loss 계산

            (
                pitcher_id,
                pitcher_discrete,
                pitcher_continuous,
                
                batter_id,
                batter_discrete,
                batter_continuous,
                
                state_discrete,
                state_continuous,
                
                pitch_mask,
                pitch,
                label,
            ) = list(map(lambda x: x.to(args.device), batch))
            model.train()
            scores, att_w, att_w2 = model(
                pitcher_id,
                pitcher_discrete,
                pitcher_continuous,
                
                batter_id,
                batter_discrete,
                batter_continuous,
                
                state_discrete,
                state_continuous,
                pitch_mask,
            )
            loss = loss_fct(scores.view(-1, 144), pitch.view(-1)) # loss는 크로스 엔트로피로 계산


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.evaluate_during_training:
                    results, _ = evaluate(args, model)
                    

                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                )
                logging_loss = tr_loss

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = "checkpoint"
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                )
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                rotate_checkpoints(args, checkpoint_prefix)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, id_list, k, prefix=""): # 예측 score 상에서 여러 조합을 이용해 다양하게 시각화하고, 다양한 metric을 계산하는 과정
    eval_output_dir = args.output_dir
    eval_dataset = DataSets(args.eval_data_file,id_list)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        dataset=eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
    )
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    loss_fct = torch.nn.CrossEntropyLoss()

    model.eval()
    total_pitch_preds1 = [] # 모델이 예측한 top_1의 아이템 인덱스를 저장하는 리스트
    total_pitch_preds3 = [] # 모델이 예측한 top_3의 아이템 인덱스를 저장하는 리스트
    total_mf_preds3 = [] # 모델이 예측한 top_3의 아이템에 대해 각각 예측 스코어를 저장하는 리스트
    total_pitch_preds5 = [] # 모델이 예측한 top_5의 아이템 인덱스를 저장하는 리스트
    total_pitch_preds10 = [] # 모델이 예측한 top_10의 아이템 인덱스를 저장하는 리스트
    total_pitch_preds20 = [] # 모델이 예측한 top_20의 아이템 인덱스를 저장하는 리스트
    total_pitch_labels = [] # 정답 구질의 인덱스를 저장하는 리스트
    total_att_w = []
    total_att_w2 = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (
            pitcher_id,
            pitcher_discrete,
            pitcher_continuous,
            
            batter_id,
            batter_discrete,
            batter_continuous,
            
            state_discrete,
            state_continuous,
            
            pitch_mask,
            pitch,
            label,
        ) = list(map(lambda x: x.to(args.device), batch))
        with torch.no_grad():
            scores, att_w, att_w2 = model(
                pitcher_id,
                pitcher_discrete,
                pitcher_continuous,
                
                batter_id,
                batter_discrete,
                batter_continuous,
                
                state_discrete,
                state_continuous,
                pitch_mask,
            )
            loss = loss_fct(scores.view(-1, 144), pitch.view(-1))

            eval_loss += loss.mean().item()
        nb_eval_steps += 1
        
        mf_preds3 = torch.topk(torch.softmax(scores.view(-1, 144), dim=1).detach().cpu(), k=3, dim=1)[0]
        pitch_preds1 = torch.topk(torch.softmax(scores.view(-1, 144), dim=1).detach().cpu(), k=1, dim=1)[1]
        pitch_preds3 = torch.topk(torch.softmax(scores.view(-1, 144), dim=1).detach().cpu(), k=3, dim=1)[1]

        pitch_preds5 = torch.topk(torch.softmax(scores.view(-1, 144), dim=1).detach().cpu(), k=5, dim=1)[1]
        pitch_preds10 = torch.topk(torch.softmax(scores.view(-1, 144), dim=1).detach().cpu(), k=10, dim=1)[1]
        pitch_preds20 = torch.topk(torch.softmax(scores.view(-1, 144), dim=1).detach().cpu(), k=20, dim=1)[1]
        
        pitch_labels = pitch.detach().cpu()
        

        total_pitch_preds1.append(pitch_preds1)
        total_pitch_preds3.append(pitch_preds3)
        total_mf_preds3.append(mf_preds3)
        
        total_pitch_preds5.append(pitch_preds5)
        total_pitch_preds10.append(pitch_preds10)        
        total_pitch_preds20.append(pitch_preds20)
        
        total_pitch_labels.append(pitch_labels)
        
        total_att_w.append(att_w)
        total_att_w2.append(att_w2)
        

    total_pitch_preds1 = torch.cat(total_pitch_preds1)
    total_pitch_preds3 = torch.cat(total_pitch_preds3)
    total_mf_preds3 = torch.cat(total_mf_preds3)
    
    total_pitch_preds5 = torch.cat(total_pitch_preds5)
    total_pitch_preds10 = torch.cat(total_pitch_preds10)
    total_pitch_preds20 = torch.cat(total_pitch_preds20)
    
    total_pitch_labels = torch.cat(total_pitch_labels)
    
    total_att_w = torch.cat(total_att_w, dim=1)
    total_att_w2 = torch.cat(total_att_w2, dim=1)
    
    
    # 모델이 예측한 아이템 인덱스 중 정답이 있는지, 몇 위로 있는지에 대한 metric을 계산 (top5, top10, top20에 대해 각각 구함)
    recalls5 = []
    mrrs5 = []
    recalls10 = []
    mrrs10 = []
    recalls20 = []
    mrrs20 = []
    
    recall5, mrr5 = metric.evaluate(total_pitch_preds5, total_pitch_labels)
    recall10, mrr10 = metric.evaluate(total_pitch_preds10, total_pitch_labels)
    recall20, mrr20 = metric.evaluate(total_pitch_preds20, total_pitch_labels)

    ### 모델이 예측한 top_1의 아이템(144개 class)에 대해, 위치정보를 제거하여 9개 class로 바꾼 뒤 전체 f1 score 와 구질별 f1 score, 그리고 confusion map을 그림
    
    for i in range(len(total_pitch_labels)):
        total_pitch_labels[i]=total_pitch_labels[i]//16
        total_pitch_preds1[i]=total_pitch_preds1[i]//16
    
    macro_f1_label_9=f1_score(
            total_pitch_labels, total_pitch_preds1, average="macro"
        )
    micro_f1_label_9=f1_score(
            total_pitch_labels, total_pitch_preds1, average="micro"
        )
    
    # 구질 별 f1-score 계산    
    label_list = list(range(9))
    
    dev_cr = classification_report(
        total_pitch_labels,
        total_pitch_preds1,
        labels=label_list,
        target_names=['CHUP', 'CURV', 'CUTT', 'FAST', 'FORK', 'KNUC', 'SINK', 'SLID', 'TWOS'],
        output_dict=True,
    )

    f1_results = [
        (l, r["f1-score"]) for i, (l, r) in enumerate(dev_cr.items()) if i < len(label_list)
    ]
    f1_log_9 = ["{} : {}".format(l, f) for l, f in f1_results]
    
    cm = confusion_matrix(total_pitch_labels, total_pitch_preds1, labels=label_list)
    
    
    
    ### 모델이 예측한 top_1의 아이템(144개 class)에 대해, 위치정보를 제거하고 구질 몇 가지를 합쳐서 3개 class로 바꾼 뒤 전체 f1 score 와 구질별 f1 score, 그리고 confusion map을 그림

    for i in range(len(total_pitch_labels)):
        if total_pitch_labels[i]==2 or total_pitch_labels[i]==3 or total_pitch_labels[i]==5 or total_pitch_labels[i]==6 or total_pitch_labels[i]==8:
            total_pitch_labels[i]=0
        elif total_pitch_labels[i]==7:
            total_pitch_labels[i]=1
        else:
            total_pitch_labels[i]=2
    
    for i in range(len(total_pitch_preds1)):
        if total_pitch_preds1[i]==2 or total_pitch_preds1[i]==3 or total_pitch_preds1[i]==5 or total_pitch_preds1[i]==6 or total_pitch_preds1[i]==8:
            total_pitch_preds1[i]=0
        elif total_pitch_preds1[i]==7:
            total_pitch_preds1[i]=1
        else:
            total_pitch_preds1[i]=2
            
        
        
    macro_f1_label_3=f1_score(
            total_pitch_labels, total_pitch_preds1, average="macro"
        )
    micro_f1_label_3=f1_score(
            total_pitch_labels, total_pitch_preds1, average="micro"
        )
    
    # 구질 별 f1-score 계산    
    label_list = list(range(3))
    
    dev_cr = classification_report(
        total_pitch_labels,
        total_pitch_preds1,
        labels=label_list,
        target_names=['Fast', 'Horizon', 'Vertical'],
        output_dict=True,
    )

    f1_results = [
        (l, r["f1-score"]) for i, (l, r) in enumerate(dev_cr.items()) if i < len(label_list)
    ]
    f1_log_3 = ["{} : {}".format(l, f) for l, f in f1_results]
    cm2 = confusion_matrix(total_pitch_labels, total_pitch_preds1, labels=label_list)

        
    eval_loss = eval_loss / nb_eval_steps
    result = {
        "pitch_macro_f1_9": macro_f1_label_9,
        "pitch_micro_f1_9": micro_f1_label_9,
        "pitch_macro_f1_3": macro_f1_label_3,
        "pitch_micro_f1_3": micro_f1_label_3,

        "loss": eval_loss,
        

        
    }
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    ####### 특정 k번째의 예시를 뽑아내기 위해, k번째 데이터에 대해 예측한 top3의 구질 및 스코어, 그리고 어텐션 스코어를 return함
    return result, f1_log_3, f1_log_9, cm, cm2, total_pitch_preds3[k], total_mf_preds3[k], total_att_w.squeeze(2).transpose(1,0)[k], total_att_w2.squeeze(2).transpose(1,0)[k], recall5, mrr5, recall10, mrr10, recall20, mrr20


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default='../data/rec_binary_data_loc_train.pkl',
        type=str,
        required=False,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default='../output_loc',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Model parameters
    parser.add_argument(
        "--n_encoder_layer", default=2,type=int, required=False, help="The number of Multimodal Transformers",
    )
    parser.add_argument(
        "--n_decoder_layer", default=2,type=int, required=False, help="The number of Multimodal Transformers",
    )
    parser.add_argument(
        "--n_concat_layer", default=2,type=int, required=False, help="The number of concat layers",
    )
    parser.add_argument(
        "--d_model", default=64,type=int, required=False, help="The dimension of self attention parameters",
    )
    parser.add_argument(
        "--nhead", default=2,type=int, required=False, help="The number of self attention head",
    )
    parser.add_argument(
        "--dim_feedforward", default=256,type=int, required=False, help="The dimension of feedforward network",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, required=False, help="Dropout probability",
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default='../data/rec_binary_data_loc_val.pkl',
        type=str, 
        help="An optional input evaluation data file to evaluate.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--train_batch_size", default=256, type=int, help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=256, type=int, help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.9, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_percent", default=0.01, type=float, help="Linear warmup over warmup_percent."
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--debug", action="store_true", help="set logging level DEBUG",
    )
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # Set seed
    set_seed(args)
    
    with open('../data/id_list.csv', newline='') as f:
        reader = csv.reader(f)
        id_list = list(reader)
    batter_len=len(set(id_list[0]))
    pitcher_len=len(set(id_list[1]))
    
    train_dataset = DataSets(args.train_data_file, id_list)
    # subtract two index variables and three categorical variables
    n_pitcher_cont = 9
    # subtract two index variables and two categorical variables
    n_batter_cont = 9
    # subtract one index variables and three categorical variables
    n_state_cont = len(train_dataset.state[0]) - 3
    
    
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
    )

    model.to(args.device)
    
    # Training
    #if args.do_train:
    if 1:
        os.makedirs(args.output_dir, exist_ok=True)
        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True)
                )
            )
            logging.getLogger("evaluation").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model.load_state_dict(torch.load(checkpoint + "/pytorch_model.bin"))
            model.to(args.device)
            result = evaluate(args, model, id_list, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)


if __name__ == "__main__":
    main()
