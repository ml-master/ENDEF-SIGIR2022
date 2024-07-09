import json
import os.path
import random

import requests
from tqdm import tqdm

prev_time = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2008", "2009",
    "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]
future_time = ['2018']

def texsmart_demo():
    str = "The Supreme Court declined on Monday to hear the appeal of Brendan Dassey , whose murder conviction was documented in the 2015 Netflix documentary series “ Making a Murderer. ” His lawyers , and a legion of supporters he gained after the popular series questioned his 2007 conviction , argued that police investigators improperly coerced his videotaped confession . Mr. Dassey , 28 , was convicted of participating with his uncle Steven Avery in the 2005 murder of Teresa Halbach , a 25-year-old photographer in Manitowoc , Wis . The 10-part Netflix series by the filmmakers Laura Ricciardi and Moira Demos suggested that police investigators unfairly questioned Mr. Dassey when he was 16 without a lawyer or parent present . He appeared scared and unaware of the gravity of his situation on camera , and his lawyers say he had a low IQ in the seventh percentile of children his age , making him susceptible to suggestion . The Supreme Court did not offer a reasoning for its decision . In a statement , Laura Nirider , a lawyer for Mr. Dassey , said , “ We will continue to fight to free Brendan Dassey . ”"
    obj = {"str": str}
    # obj = {"str": "他在看流浪地球。"}
    req_str = json.dumps(obj).encode()

    url = "https://texsmart.qq.com/api"
    r = requests.post(url, data=req_str)
    r.encoding = "utf-8"
    print(r.text)
    #print(json.loads(r.text))

def extract_entity(text):
    obj = {
        "str": text,
        "options":
            {
                "input_spec": {"lang": "auto"},
                "word_seg": {"enable": False},
                "pos_tagging": {"enable": False},
                "ner": {"enable": True, "alg": "fine.std"},
                "syntactic_parsing": {"enable": False},
                "srl": {"enable": False},
                "text_cat": {"enable": False},
            }
    }
    req_str = json.dumps(obj).encode()
    url = "https://texsmart.qq.com/api"
    r = requests.post(url, data=req_str)
    r.encoding = "utf-8"
    try:
        result = json.loads(r.text)['entity_list']
    except:
        print(r.text)
        result = None
    return result

def filtering_data(data):
    p_data = []
    f_data = []
    for elem in data:
        if elem['time'].strip()[:4] in prev_time:
            p_data.append(elem)
        if elem['time'].strip()[:4] in future_time:
            f_data.append(elem)
    return p_data, f_data

def split_dataset(data):
    p_data, f_data = filtering_data(data)
    p_num = len(p_data)
    f_num = len(f_data)
    total = p_num + f_num
    training_data = p_data
    valid_data = []
    test_data = []

    val_test_num = int(total * 0.2)
    split_array = [0] * val_test_num + [1] * val_test_num + [2] * (f_num - 2 * val_test_num)
    random.shuffle(split_array)

    for i, elem in enumerate(f_data):
        if split_array[i] == 0:
            valid_data.append(elem)
        elif split_array[i] == 1:
            test_data.append(elem)
        else:
            training_data.append(elem)
    return training_data, valid_data, test_data

if __name__ == '__main__':
    data_dir = 'data/GossipCop-LLM-Data-examples/'
    data_path = data_dir + 'gossipcop_v3_origin.json'
    # data_path = data_dir + 'gossipcop_v3_origin-mini.json'
    output_path = data_dir + 'format_data.json'
    # output_path = data_dir + 'format_data-mini.json'

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            format_data = json.load(file)
    else:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        format_data = []
        invalid_num = 0     # 没有时间数据的数量
        for key, value in tqdm(data.items(), total=len(data)):
            text = value['text']
            entity_list = extract_entity(text)
            if entity_list == None:
                invalid_num += 1
                continue
            label = 1 if value['label'] == 'fake' else 0
            try:
                time = value['meta_data']['article']['published_time']
            except:
                invalid_num += 1
                continue

            elem = {
                "content": text,
                "label": label,
                "time": time,
                "entity_list": entity_list
            }
            format_data.append(elem)

        format_json_string = json.dumps(format_data, indent=4, ensure_ascii=False)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(format_json_string)

        print(f"valid_num: {len(format_data)}\n"
              f"invalid_num: {invalid_num}")

    training_data, valid_data, test_data = split_dataset(format_data)
    training_json_string = json.dumps(training_data, indent=4, ensure_ascii=False)
    valid_json_string = json.dumps(valid_data, indent=4, ensure_ascii=False)
    test_json_string = json.dumps(test_data, indent=4, ensure_ascii=False)
    train_path = data_dir + 'train.json'
    val_path = data_dir + 'val.json'
    test_path = data_dir + 'test.json'
    with open(train_path, 'w', encoding='utf-8') as f1, \
            open(val_path, 'w', encoding='utf-8') as f2, \
            open(test_path, 'w', encoding='utf-8') as f3:
        f1.write(training_json_string)
        f2.write(valid_json_string)
        f3.write(test_json_string)

    print(f'train instances num: {len(training_data)}\n'
          f'valid instances num: {len(valid_data)}\n'
          f'test instances num: {len(test_data)}')
    print('end')

