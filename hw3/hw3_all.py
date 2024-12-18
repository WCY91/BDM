import re
import random
from collections import defaultdict 
import json
import time
# ## 資料預處理

def extract_train_body_contents(file_path):
    # 讀取文件內容
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        file_content = file.read()

    # 使用正則表達式查找符合要求的<REUTERS>標籤
    pattern = re.compile(r'<REUTERS[^>]*TOPICS="YES" LEWISSPLIT="T[^>]*>.*?</REUTERS>', re.DOTALL)
    train_reuters_tags = pattern.findall(file_content)
    
    # 提取每個標籤中的<BODY>內容
    train_body_contents = []
    train_body_label_topic_list = []
    for reuters_tag in train_reuters_tags:
        topic_pattern = re.compile(r'<TOPICS>(.*?)</TOPICS>') #use to calculate the lsh eval score
        body_pattern = re.compile(r'<BODY>(.*?)</BODY>', re.DOTALL)
        body_content = body_pattern.search(reuters_tag)
        topic_content = topic_pattern.search(reuters_tag)
        cleaned_content = []
        if body_content and topic_content:
            # 移除非單詞字符
            cleaned_content = body_content.group(1).replace("\n", ' ')
            # 轉換為小寫
            cleaned_content = cleaned_content.lower()
            train_body_contents.append(cleaned_content)

            # train_body_contents.append(body_content.group(1).strip())
            d_pattern = re.compile(r'<D>(.*?)</D>') # use to catch the topic list
            all_topics = []
            d_topics = d_pattern.findall(topic_content.group(1))
            all_topics.append(d_topics)
            train_body_label_topic_list.append(all_topics)
    # print(len(train_body_label_topic_list))
    # print(len(train_body_contents))

    
    return train_body_contents,train_body_label_topic_list

data = []
topic = []
for i in ('000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019', '020', '021'):
    file = "./reuters+21578+text+categorization+collection/reuters21578/reut2-"+i+'.sgm'
    #print("Processing: " + file)
    result = extract_train_body_contents(file)
    data.extend(result[0])
    topic.extend(result[1])

#print("data = ", data)


## k-shingles
def get_shingles(text, k):
    """生成k-shingles"""
    return set(text[i:i+k] for i in range(len(text) - k + 1))


# 自定義的k值
# k = 3
# k = input("Enter the k value: (recommend 3)")
try:
    k = int(input("Enter the k value (recommend 3): "))
except ValueError:
    print("Please enter a valid integer.")

# 生成每個data的k-shingles
all_shingles = set()
shingles_per_document = []
for doc in data:
    shingles = get_shingles(doc, k)
    all_shingles.update(shingles)
    shingles_per_document.append(shingles)

# 建shingles矩陣
shingles_list = sorted(list(all_shingles))

matrix = []
for shingle in shingles_list:
    row = [1 if shingle in doc_shingles else 0 for doc_shingles in shingles_per_document]
    matrix.append(row)



def get_minhash_functions(num_functions, num_rows):
    # 生成minhash函数的列表
    def create_hash_function():
        a, b = random.randint(1, 10 * num_rows), random.randint(0, 10 * num_rows)
        return lambda x: (a * x + b) % num_rows

    return [create_hash_function() for _ in range(num_functions)]


def get_minhash_signature(feature_matrix, num_functions): 
    num_rows = len(feature_matrix)
    minhash_funcs = get_minhash_functions(num_functions, num_rows)
    # print(f"in minhash function {len(feature_matrix)} and {len(feature_matrix[1])}")
    # 初始化signatures matrix，初始值設為無限大
    signatures = [[float('inf')] * len(feature_matrix[0]) for _ in range(num_functions)]

    for row_idx, row in enumerate(feature_matrix):
        for col_idx, value in enumerate(row):
            if value == 1:
                for func_idx, func in enumerate(minhash_funcs):
                    signatures[func_idx][col_idx] = min(signatures[func_idx][col_idx], func(row_idx))
    
    return signatures

try:
    num_functions = int(input("Enter the number of functions (recommend 100): "))
except ValueError:
    print("Please enter a valid integer.")
minhash_signatures = get_minhash_signature(matrix, num_functions)


#hw3-3
s = 0.8 #s is a similar threshold 

#LSH 是將min-hash matrix 其中r個組成一個band 而將matrix分成b個band並將每個band中的vector(r個元素) 去hash進k個buckets
#如何判斷是否為候選的pair column 當有至少一個band hash進同一個bucket
#Tune b and r to catch most similar pairs, but few non-similar pairs
#k盡量大一點 > 10000
try:
    k=100
    k = int(input("Enter the number of buckets you want "))
    band = 20
    band = int(input("Enter the band number band need to bigger than num of hash functions  "))
except:
    k=100
    band = 20


def lsh(band=20, k=100, matrix=[], num_functions=100):
    r = num_functions // band
    result_candidate_pair = defaultdict(set)
    band_buckets = [defaultdict(set) for _ in range(band)]

    #將個別doc的band去hash到bucket中
    for doc in range(len(matrix[0])):  
        for band_id in range(band):  
            doc_band_r_vector = tuple(row[doc] for row in matrix[band_id * r: band_id * r + r])
            hash_location_key = hash(doc_band_r_vector) % k
            band_buckets[band_id][hash_location_key].add(doc)
    print(len(band_buckets))

    #找candidate pair
    for band_id in range(band):
        for bucket_docs in band_buckets[band_id].values():
            if len(bucket_docs) > 1:
                for doc1 in bucket_docs:
                    for doc2 in bucket_docs:
                        if doc1 < doc2: 
                            result_candidate_pair[doc1].add(doc2)
                            result_candidate_pair[doc2].add(doc1) 
    
    print(len(result_candidate_pair))

    return result_candidate_pair

    
# #evaluate lsh result -------> hw3-4
def preprocess_doc_topic_pairs(doc_topic_pairs):
    return {i: set(tuple(topic) if isinstance(topic, list) else topic for topic in topics)
            for i, topics in enumerate(doc_topic_pairs)}

def evaluate_sim(lsh_pairs, doc_topic_pairs):
    doc_topic_pairs = preprocess_doc_topic_pairs(doc_topic_pairs)

    confuse_matrix = defaultdict(int)
    num_docs = len(doc_topic_pairs)

    for doc_id, candidate_set in lsh_pairs.items():
        for candidate_doc_id in candidate_set:
            if doc_topic_pairs[doc_id] & doc_topic_pairs[candidate_doc_id]:
                confuse_matrix['tp'] += 1
            else:
                confuse_matrix['fp'] += 1

    all_docs = set(range(num_docs))
    for doc_id in all_docs:
        non_candidates = all_docs - set(lsh_pairs.get(doc_id, set())) - {doc_id}
        for other_doc in non_candidates:
            if doc_topic_pairs[doc_id] & doc_topic_pairs[other_doc]:
                confuse_matrix['fn'] += 1
            else:
                confuse_matrix['tn'] += 1
    
    confuse_matrix['tp_percentage'] = float(confuse_matrix['tp']*1.0 / (confuse_matrix['tp'] + confuse_matrix['fp']))
    confuse_matrix['tn_percentage'] = float(confuse_matrix['tn']*1.0 / (confuse_matrix['tn'] + confuse_matrix['fn']))
    confuse_matrix['fp_percentage'] = float(confuse_matrix['fp']*1.0 / (confuse_matrix['tp'] + confuse_matrix['fp']))
    confuse_matrix['fn_percentage'] = float(confuse_matrix['fn']*1.0 / (confuse_matrix['tn'] + confuse_matrix['fn']))

    return dict(confuse_matrix)


def write_candidates_to_file(candidate_pairs, file_name="candidate_pairs.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        for bucket_key, docs in candidate_pairs.items():
            file.write(f"Doc {bucket_key}: {docs}\n")

def write_eval_result_to_file(eval_result, file_name="evaluation_result.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        for key, values in eval_result.items():
            file.write(f" {key}: {values}\n")
        json.dump(eval_result, file, ensure_ascii=False, indent=4)

candidate_pair = lsh(band,k,minhash_signatures)
print(len(candidate_pair))
print(len(candidate_pair[0]))

eval_result = evaluate_sim(candidate_pair,topic)
print(f"evaluate the lsh result the tp:{eval_result['tp_percentage']} , fp:{eval_result['fp_percentage']} , tn:{eval_result['tn_percentage']}, fn: {eval_result['fn_percentage']}")

write_candidates_to_file(candidate_pair, "candidate_pairs.txt")
write_eval_result_to_file(eval_result, "evaluation_result.txt")



     

#hw3-5 LSH + KNN
def compare_knn_distance(query,compare_doc,min_hash_matrix):
    query_signature = [row[query] for row in min_hash_matrix]
    compare_signature = [row[compare_doc] for row in min_hash_matrix]
    return sum(1 for q, c in zip(query_signature, compare_signature) if q == c) / len(query_signature) ##jaccard 


def knn_lsh(query ,lsh_pairs,k,min_hash_matrix): #query = you want to query doc (doc id )
    all_query_candidates = lsh_pairs[query]
    distance = []
    for i in range(len(all_query_candidates)):
        if i == query:continue
        distance.append((i,compare_knn_distance(query,i,min_hash_matrix))) #裡面會存所有的相似度數值

    distance = sorted(distance, key=lambda x: x[1], reverse=True) #相似度越大越好 所以大到小
    return distance[:k]

def knn_linear(query ,k,data,min_hash_matrix):
    distance = []
    for i in range(len(data)):
        if i== query: continue
        distance.append((i,compare_knn_distance(query,i,min_hash_matrix))) 
    
    distance = sorted(distance, key=lambda x: x[1], reverse=True)
    return distance[:k]

def write_knn_compare_result(knn_linear_result,knn_lsh_result,exec_knn_linear,exec_knn_lsh, file_name="knn_compare_result.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(f"the knn_linear result is :\n")
        for doc_sim in knn_linear_result:
            file.write(f" the doc is {doc_sim[0]}: the jaccard with the doc id 0 sim is :{doc_sim[1]}\n")
        file.write(f"the knn_linear exec time is {exec_knn_linear}\n\n")

        file.write(f"the knn_lsh result is :\n")
        for doc_sim in knn_lsh_result:
            file.write(f" the doc is {doc_sim[0]}: the jaccard with the doc id 0 sim is :{doc_sim[1]}\n")
        file.write(f"the knn_lsh exec time is {exec_knn_lsh}\n\n")

time_knn_lsh = 0
time_knn_linear = 0

start_time = time.time()
knn_linear_result = knn_linear(0,7,data,minhash_signatures)
end_time = time.time()
time_knn_linear = end_time - start_time

start_time = time.time()
knn_lsh_result = knn_lsh(0,candidate_pair,7,minhash_signatures)
end_time = time.time()
time_knn_lsh = end_time - start_time

write_knn_compare_result(knn_linear_result,knn_lsh_result,time_knn_linear,time_knn_lsh)








