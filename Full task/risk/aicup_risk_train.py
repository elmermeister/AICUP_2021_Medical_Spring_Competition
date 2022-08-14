import pandas as pd
import numpy as np
import os,re,time,datetime
import torch


#==&==


#參數設定
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--GPU",
                    nargs = '?',
                    type = int,
                    default = 0 ,
                    help = "GPU選擇張數"
                    )
                    
parser.add_argument("--MODEL_NAME",
                    nargs = '?',
                    type = str,
                    default =  'hfl/chinese-roberta-wwm-ext-large',
                    help = "讀取的pre-train模型參數及架構"
                    )

parser.add_argument("--SAVE_STATE_NAME",
                    nargs = '?',
                    type = str,
                    default =  'risk_model.pt',
                    help = "存取的模型參數名稱"
                    )

parser.add_argument("--MODEL_SAVE_PATH",
                    nargs = '?',
                    type = str,
                    default = 'aicup_risk_output' ,
                    help = "模型存取位置"
                    )

parser.add_argument("--BATCH_SIZE",
                    nargs = '?',
                    type = int,
                    default = 8,
                    help = "Training data 的 batch size大小"
                    )

parser.add_argument("--EPOCHS",
                    nargs = '?',
                    type = int,
                    default =  4,
                    help = "訓練次數epochs的調整"
                    )

parser.add_argument("--LR",
                    nargs = '?',
                    type = float,
                    default = 2e-5 ,
                    help = ""
                    )

parser.add_argument("--MAX_SEQ_LENGTH",
                    nargs = '?',
                    type = int,
                    default =  512,
                    help = "RoBERTa model中的最大輸入大小調整"
                    )

args = parser.parse_args()

GPU = args.GPU
MODEL_NAME = args.MODEL_NAME
SAVE_STATE_NAME = args.SAVE_STATE_NAME
MODEL_SAVE_PATH = args.MODEL_SAVE_PATH
BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS
LR = args.LR
MAX_SEQ_LENGTH = args.MAX_SEQ_LENGTH


#==&==

#設定gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#讀取訓練資料
train_path = os.path.join('aicup_risk_data','Train_risk_classification_ans.csv')
train_df = pd.read_csv(train_path)
train_df = train_df[['article_id','text','label']]

print(f'number of training data : { len(train_df) }')
print()


#==&==


from torch.utils.data import Dataset
from transformers import BertTokenizer
# 載入 BERT 分詞器
print(f'Loading {MODEL_NAME} tokenizer...')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 處理資料
class RiskData(Dataset):
    def __init__(self,data):
        self.df = self.split_func(data)
        
    def split_func(self,data):
        
        #記算字的數量
        def count_words(sentence_list):
            c = int()
            for i in sentence_list[int(len(sentence_list)*2/3):]:
                c += len(i)
            return c

        df_list = list()

        for article_id,dialogue,label in data.values:
            #依照 speaker 斷開
            split_list = re.split(r'([\u4E00-\u9FFFa-zA-Z0-9]+[：])',dialogue)
            split_list.pop(0)
            sentence_list = ["".join(i) for i in zip(split_list[0::2],split_list[1::2])]
            #擷取 個管師 or 醫師 部分
            sentence_list = [s for s in sentence_list if (('個管師' in s[:3]) or ('醫師' in s[:3]))]
            #刪除 '人：'
            sentence_list = [ re.sub(r'([\u4E00-\u9FFFa-zA-Z0-9]+[：])',"",s) for s in sentence_list]
            #過濾 字數少於 10 的句子
            sentence_list = [ s for s in sentence_list if len(s) > 10]
  
            #取中間
            if count_words(sentence_list) > 500 :          
                key,i,j= True,0,1
                while (key):
                    if (count_words(sentence_list[i:-j]) < 500):
                        sentence_list = sentence_list[i:-j]
                        key = False
                    if(i==j):
                        i += 1
                    else:
                        j += 1
                        
            sentence = " ".join(sentence_list)
                        
            encoded_dict = tokenizer.encode_plus(
                        sentence,                      # 輸入文字
                        add_special_tokens = True, # 新增 '[CLS]' 和 '[SEP]'
                        max_length = MAX_SEQ_LENGTH,           # 填充 & 截斷長度
                        pad_to_max_length = True,
                        return_attention_mask = True,   # 返回 attn. masks.
                        return_tensors = 'pt',     # 返回 pytorch tensors 格式的資料
                   )
            #reshape 資料
            inp_id = torch.reshape(encoded_dict['input_ids'],(encoded_dict['input_ids'].size(1),))
            att_mask = torch.reshape(encoded_dict['attention_mask'],(encoded_dict['attention_mask'].size(1),))
            token_ids = torch.reshape(encoded_dict['token_type_ids'],(encoded_dict['token_type_ids'].size(1),))
            #塞入list中
            df_list.append( [torch.tensor(article_id), sentence , inp_id , token_ids , att_mask , torch.tensor(int(label)) ] )
        #將list 轉成 df 
        df = pd.DataFrame(data = df_list,columns=['article_id','sentence','input_ids','token_type_ids','attention_mask','label'])
        return df

    def __getitem__(self,index):
        return self.df['article_id'][index],self.df['input_ids'][index],self.df['attention_mask'][index],self.df['label'][index]

    def __len__(self):
        return len(self.df)
    
    def get_original(self,index):
        return self.df['sentence'][index],self.df['label'][index]


#==&==


from torch.utils.data import  RandomSampler, SequentialSampler,DataLoader
from torch.utils.data.dataset import random_split

#計算 train dev size 
train_size = int(0.9 * len(train_df))
dev_size = len(train_df) - train_size
print(f'dataset: train_size = {train_size}')
print(f'dataset: dev_size = {dev_size}')

def get_data():
    
    # 按照資料大小隨機拆分訓練集和測試集
    train_dataset, dev_dataset = random_split(RiskData(train_df), [train_size, dev_size])

    # 為訓練和驗證集建立 Dataloader，對訓練樣本隨機洗牌
    train_dataloader = DataLoader(
                train_dataset,  # 訓練樣本
                sampler = RandomSampler(train_dataset), # 隨機小批量
                batch_size = BATCH_SIZE # 以小批量進行訓練
            )

    # 驗證集不需要隨機化，這裡順序讀取就好
    validation_dataloader = DataLoader(
                dev_dataset, # 驗證樣本
                sampler = SequentialSampler(dev_dataset), # 順序選取小批量
                batch_size = BATCH_SIZE 
            )
    return train_dataloader,validation_dataloader


#==&==


#建立模型
from transformers import AdamW , BertForSequenceClassification

def create_model():
    print(f'load model : {MODEL_NAME}')

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels = 2, # 分類數
        output_attentions = False, # 模型是否返回 attentions weights.
        output_hidden_states = False, # 模型是否返回所有隱層狀態.
    )

    # 將模型架構輸出至risk_model_structure.txt
    params = list(model.named_parameters()) 
    with open(os.path.join(MODEL_SAVE_PATH,'risk_model_structure.txt'),'w',encoding='utf-8') as f:
        for i in params:
            f.write(i[0]+'\n')
    
    return model


#==&==


from transformers import get_linear_schedule_with_warmup


def create_opt(model):
    optimizer = AdamW(model.parameters(),
                  lr = LR,
                  eps = 1e-8 
                  )
    return optimizer

def create_lr_scheduler(opt,train_dataloader):
    # 總的訓練樣本數
    total_steps = len(train_dataloader) * EPOCHS
    # 建立學習率排程器
    scheduler = get_linear_schedule_with_warmup(opt, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    return scheduler


#==&==


# 根據預測結果和標籤資料來計算準確率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    # 四捨五入到最近的秒
    elapsed_rounded = int( round((elapsed)) )
    # 格式化為 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

from sklearn.metrics import classification_report,f1_score

def get_report(y_test,predictions):
    target_names = ['0(無風險)','1(有風險)']
    print(classification_report(y_test,predictions,target_names=target_names))

#==&==


def train(model,train_dataloader,validation_dataloader,optimizer,scheduler):
    # 儲存訓練和評估的 loss、準確率、訓練時長等統計指標
    training_stats = []

    # 統計整個訓練時長
    total_t0 = time.time()

    for epoch_i in range(0, EPOCHS):
        
        # ========================================
        #               Training
        # ========================================
        

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        # 統計單次 epoch 的訓練時間
        t0 = time.time()

        # 重置每次 epoch 的訓練總 loss
        total_train_loss = 0

        # 將模型設定為訓練模式
        model.train()

        # 訓練集小批量迭代
        for step, batch in enumerate(train_dataloader):

            # 每經過40次迭代，就輸出進度資訊
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 準備輸入資料，並將其拷貝到 gpu 中
            b_input_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            # 梯度清空
            model.zero_grad()        

            # 前向傳播
            m = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
            loss, logits = m['loss'],m['logits']
            # 累加 loss
            total_train_loss += loss.item()

            # 反向傳播
            loss.backward()

            # 梯度裁剪，避免出現梯度爆炸情況
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新引數
            optimizer.step()

            # 更新學習率
            scheduler.step()

        # 平均訓練誤差
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # 單次 epoch 的訓練時長
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # 完成一次 epoch 訓練後，就對該模型的效能進行驗證

        print("")
        print("Running Validation...")

        t0 = time.time()

        # 設定模型為評估模式
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        label_list = np.array([])
        predict_list = np.array([])
        
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # 將輸入資料載入到 gpu 中
            b_input_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            # 評估的時候不需要更新引數、計算梯度
            with torch.no_grad():        
                m = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels)
                loss, logits = m['loss'],m['logits']
            # 累加 loss
            total_eval_loss += loss.item()

            # 將預測結果和 labels 載入到 cpu 中計算
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            l = np.argmax(logits, axis=1).flatten()
            predict_list = np.append(predict_list,l)
            label_list = np.append(label_list,label_ids)


            # 計算準確率
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # 列印本次 epoch 的準確率
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # 統計本次 epoch 的 loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # 統計本次評估的時長
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # 記錄本次 epoch 的所有統計資訊
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        get_report(label_list,predict_list)

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # 保留 2 位小數
    pd.set_option('precision', 2)

    # 載入訓練統計到 DataFrame 中
    df_stats = pd.DataFrame(data=training_stats)
    
    # 使用 epoch 值作為每行的索引
    df_stats = df_stats.set_index('epoch')

    print(df_stats)

    return model


#==&==


# main


#取得資料
train_dataloader,validation_dataloader = get_data()

#建立模型
model = create_model()
model.cuda()

#建立opt
optimizer = create_opt(model)
#建立scheduler
scheduler = create_lr_scheduler(optimizer,train_dataloader)

#train
model = train(model,train_dataloader,validation_dataloader,optimizer,scheduler)

#save model
path = os.path.join(MODEL_SAVE_PATH,SAVE_STATE_NAME)
torch.save(model.state_dict(),path)


