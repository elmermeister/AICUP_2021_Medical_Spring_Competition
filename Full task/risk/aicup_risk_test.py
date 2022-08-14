import pandas as pd
import numpy as np
import os,re
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

parser.add_argument("--LOAD_STATE_NAME",
                    nargs = '?',
                    type = str,
                    default =  'risk_model.pt',
                    help = "存取的模型參數名稱"
                    )

parser.add_argument("--BATCH_SIZE",
                    nargs = '?',
                    type = int,
                    default = 8,
                    help = "Test data 的 batch size大小"
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
LOAD_STATE_NAME = args.LOAD_STATE_NAME
BATCH_SIZE = args.BATCH_SIZE
MAX_SEQ_LENGTH = args.MAX_SEQ_LENGTH


#==&==


#設定gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#讀取測試資料
test_path = os.path.join('aicup_risk_data','Test_risk_classification.csv')
test_df = pd.read_csv(test_path)
test_df = test_df[['article_id','text','label']]
test_df['label'] = 0

print(f'number of develop data : { len(test_df) }')
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
            split_list = re.split(r'([\u4E00-\u9FFFa-zA-Z0-9]+[：])',dialogue)
            split_list.pop(0)
            sentence_list = ["".join(i) for i in zip(split_list[0::2],split_list[1::2])]
            #擷取 個管師 or 醫師 部分
            sentence_list = [s for s in sentence_list if (('個管師' in s[:3]) or ('醫師' in s[:3]))]
            #刪除 人：
            sentence_list = [ re.sub(r'([\u4E00-\u9FFFa-zA-Z0-9]+[：])',"",s) for s in sentence_list]
            #過濾 字數少於 10 的句子
            sentence_list = [ s for s in sentence_list if len(s) > 10]

            #選頭尾
            if count_words(sentence_list) > 500 :          
                key,i,j= True,0,1
                while (key):
                    if (count_words(sentence_list[i:-j])<500):
                        sentence_list = sentence_list[i:-j]
                        key = False
                    if(i==j):
                        i += 1
                    else:
                        j += 1
            sentence = "".join(sentence_list)
            
            encoded_dict = tokenizer.encode_plus(
                        sentence,                      # 輸入文字
                        add_special_tokens = True, # 新增 '[CLS]' 和 '[SEP]'
                        max_length = MAX_SEQ_LENGTH,  # 填充 & 截斷長度
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


#建立模型
from transformers import BertForSequenceClassification

def create_model():
    print(f'load model : {MODEL_NAME}')

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels = 2, # 分類數
        output_attentions = False, # 模型是否返回 attentions weights
        output_hidden_states = False, # 模型是否返回所有隱層狀態
    )
    
    return model


#==&==


from torch.utils.data import SequentialSampler,DataLoader

# 計算機率
def get_pred_probability(preds):
    pred_softmax = torch.nn.functional.softmax(preds)
    pred_softmax = pred_softmax.numpy()
    pred_softmax = np.around(pred_softmax, decimals=2)
    return pred_softmax

def test():

    #建立模型
    model = create_model()

    #載入訓練好的參數
    par_state = torch.load( os.path.join("aicup_risk_model_state",LOAD_STATE_NAME ) )
    #將模型引用參數
    model.load_state_dict(par_state, strict=False)
    model.cuda()

    #建立資料
    test_dataset = RiskData(test_df)
    test_dataloader = DataLoader(
                test_dataset, # 驗證樣本
                sampler = SequentialSampler(test_dataset), # 順序選取小批量
                batch_size = BATCH_SIZE 
            )

    # 存放預測機率
    pred_probability_list = list()

    print("")
    print("Running Test...")

    # 設定模型為評估模式
    model.eval()

    # Test data for one epoch
    for batch in test_dataloader:

        # 將輸入資料載入到 gpu 中
        b_article_id = batch[0]
        b_input_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # 評估的時候不需要更新引數、計算梯度
        with torch.no_grad():        
            m = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels)
            logits = m['logits']


        # 將預測結果載入到 cpu 中計算
        logits = logits.detach().cpu().numpy()

        # 獲得機率
        pred_probability = get_pred_probability(torch.from_numpy(logits))
        
        for name,probability in zip(b_article_id,pred_probability):
            pred_probability_list.append([int(name),probability[1]])

    #輸出檔案
    df_probability = pd.DataFrame(pred_probability_list,columns=['article_id','probability'])
    df_probability.to_csv(os.path.join('aicup_risk_output','decision.csv'),index=False)

    print('test finish')


#==&==

#test
test()

