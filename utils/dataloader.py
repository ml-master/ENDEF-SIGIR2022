import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

label_dict = {
    "real": 0,
    "fake": 1
}

category_dict = {
    "2000": 0,
    "2001": 0,
    "2002": 0,
    "2003": 0,
    "2005": 0,
    "2004": 0,
    "2006": 0,
    "2007": 0,
    "2008": 0,
    "2009": 0,
    "2010": 0,
    "2011": 0,
    "2012": 0,
    "2013": 0,
    "2014": 0,
    "2015": 0,
    "2016": 0,
    "2017": 1,
    "2018": 2
}

def word2input(texts, max_len):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('/home/minglq/Data/project/FakeNewsDetection/ENDEF-SIGIR2022/ENDEF_en/huggingface/bert-base-uncased', local_files_only=True)
    token_ids = []
    for i, text in tqdm(enumerate(texts), total=len(texts), desc='tokenize: '):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def data_augment(content, entity_list, aug_prob):
    entity_content = []
    random_num = random.randint(1,100)
    if random_num <= 50:
        for item in entity_list:
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                # content = content.replace(item["entity"], '[MASK]')
                content = content.replace(item["str"], '[MASK]')
            elif random_num <= int(2 * aug_prob * 100):
                # content = content.replace(item["entity"], '')
                content = content.replace(item["str"], '')
            else:
                # entity_content.append(item["entity"])
                entity_content.append(item["str"])
        entity_content = ' [SEP] '.join(entity_content)
    else:
        content = list(nltk.word_tokenize(content))
        for index in range(len(content) - 1, -1, -1):
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                del content[index]
            elif random_num <= int(2 * aug_prob * 100):
                content[index] = '[MASK]'
        content = ' '.join(content)
        entity_content = get_entity(entity_list)

    return content, entity_content

def get_entity(entity_list):
    entity_content = []
    for item in entity_list:
        # entity_content.append(item["entity"])
        entity_content.append(item["str"])
    entity_content = ' [SEP] '.join(entity_content)
    return entity_content

def get_dataloader(path, max_len, batch_size, shuffle, use_endef, aug_prob):
    data_list = json.load(open(path, 'r',encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content','label'))
    for item in data_list:
        tmp_data = {}
        if shuffle == True and use_endef == True:
            tmp_data['content'], tmp_data['entity'] = data_augment(item['content'], item['entity_list'], aug_prob)
        else:
            tmp_data['content'] = item['content']
            tmp_data['entity'] = get_entity(item['entity_list'])
        tmp_data['label'] = item['label']
        # tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        tmp_data['year'] = item['time'].strip()[:4]
        # df_data = df_data.append(tmp_data, ignore_index=True)
        df_data = pd.concat([df_data, pd.DataFrame([tmp_data])], ignore_index=True)
    # emotion = np.load(path.replace('.json', '_emo.npy')).astype('float32')
    # emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    entity_content = df_data['entity'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())
    year = torch.tensor(df_data['year'].apply(lambda c: category_dict[c]).astype(int).to_numpy())
    content_token_ids, content_masks = word2input(content, max_len)
    entity_token_ids, entity_masks = word2input(entity_content, 50)
    dataset = TensorDataset(content_token_ids,
                            content_masks,
                            entity_token_ids,
                            entity_masks,
                            label,
                            year,
                            # emotion
                            )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader