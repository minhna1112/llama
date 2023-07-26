from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import pandas as pd
import torch
import os

dirname = os.path.dirname(os.path.abspath(__file__))

class NLLB():
    def __init__(self, model_name) -> None:
        # if not os.path.exists(model_name):
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            print("Saving model to ", model_name)
            self.model.save_pretrained(model_name)
        
        if torch.cuda.is_available():
            print("Using this GPU device for prediction : ", torch.cuda.get_device_name(0), "....")
            device = torch.device('cuda')
        else:
            print("Using CPU for prediction ......")
            device = torch.device('cpu')
        self.model.to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        # self.load_checkpoint(model_name)
        self.langcodes = pd.read_csv(os.path.join(dirname,'flores200.csv'), sep='|')

        
    #def to(self, device: str):
    #    self.device = device       
        #print(device)
    #    self.model.to(device)

    def predict(self, input_ids, attention_mask, forced_bos_token_id, max_length):
        print('BOS id', forced_bos_token_id.shape)        
        print("input ids", input_ids.shape)
        print('max length', max_length.shape)
        print('attention_mask', attention_mask.shape)
        #print('device', self.device)
        input_ids = torch.from_numpy(input_ids)
        input_ids = input_ids.to(self.device)
        attention_mask = torch.from_numpy(attention_mask)
        attention_mask = attention_mask.to(self.device)
        translated_tokens = self.model.generate(
            input_ids= input_ids,
            attention_mask = attention_mask,
            forced_bos_token_id=forced_bos_token_id[0], 
            max_length=max_length[0],
        )
        #print(translated_tokens.device)
        return translated_tokens

# if __name__ =='__main__':

    # article = ['Create a new QRinput list', 'This is the main entry point for the QRinput library']
    # nllb = NLLB('../checkpoints/nllb-200-distilled-600M')
    # nllb.to('cuda')
    # out = nllb.predict(article, lang='German')

    # # print(model.config)
    # print(out)
