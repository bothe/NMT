import pickle
from Utils import check_valid_path

def load_word_dic(wordDicPath) :
    check_valid_path(wordDicPath)
        
    print("word dictionary 불러오는 중....")
    with wordDicPath.open("rb") as fp :
        word2id = pickle.load(fp)
    id2word = {v : k for k, v in list(word2id.items())}
    vocab_size = len(word2id)
    
    return word2id, id2word, vocab_size

def id_to_word(ID_list, id2word) :
    return [id2word[num] for num in ID_list]

def word_to_id(token_list, word2id, add_start_tag, add_end_tag, error_when_no_token) :
    num_list = [word2id["<START>"]] if add_start_tag else []
    for token in token_list :
        if token in word2id :
            num_list.append(word2id[token])
        else :
            if error_when_no_token :
                raise Exception("\n[ {} ]은 dictionary에 존재하지 않는 token입니다.\n".format(token))
            num_list.append(word2id["<UNK>"])

    if add_end_tag :
        num_list.append(word2id["<END>"])
        
    return num_list

def get_dataset(word2id, dataset_path, add_start_tag, add_end_tag, error_when_no_token) :
    # dataset에는 이미 전처리가 완료된 token이 저장되어 있음
    check_valid_path(dataset_path)
    
    with dataset_path.open("r", encoding = "utf-8") as fp :
        print("[ {} ] dataset 불러오는 중...".format(dataset_path))
        token_lists = [word_to_id(line.split(" "), word2id, add_start_tag, add_end_tag, error_when_no_token) 
            for line in [line.strip() for line in fp.readlines()]]

    num_data = len(token_lists)
    
    return token_lists, num_data