from pycocotools.coco import COCO
import numpy as np
from scipy.sparse import dok_matrix

# 初始化COCO API
ann_file = 'E:\\dataset\\coco\\annotations\\instances_train2017.json'
coco = COCO(ann_file)

# 获取所有类别和对应的id
cats = coco.loadCats(coco.getCatIds())
cat_id_map = {cat['id']: cat['name'] for cat in cats}
num_cats = len(cat_id_map)

# 统计每个类别出现在哪些图片中
cat_img_ids = [[] for _ in range(num_cats)]
img_ids = coco.getImgIds()
for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id >80 or cat_id<1:
            # ann['category_id']
            print(cat_id)
            continue
        cat_img_ids[cat_id-1].append(img_id)

# 统计每对物体出现的次数
co_occurrence_matrix = dok_matrix((num_cats, num_cats), dtype=np.int32)
for cat_i in range(num_cats):
    for cat_j in range(cat_i+1, num_cats):
        img_ids_i = set(cat_img_ids[cat_i])
        img_ids_j = set(cat_img_ids[cat_j])
        num_co_occurrence = len(img_ids_i.intersection(img_ids_j))
        co_occurrence_matrix[cat_i, cat_j] = num_co_occurrence
        co_occurrence_matrix[cat_j, cat_i] = num_co_occurrence
# print(co_occurrence_matrix.maximum())
# 归一化处理，得到邻接矩阵
degree_matrix = np.array(co_occurrence_matrix.sum(axis=1)).reshape(-1)
degree_matrix = np.sqrt(degree_matrix) + 1e-8
normalized_matrix = co_occurrence_matrix.multiply(1 / np.outer(degree_matrix, degree_matrix))

# # 保存矩阵到pkl中
import pickle
with open('coco_adj.pkl', 'wb') as f:
    pickle.dump(normalized_matrix, f)
import torch
import torch.nn as nn
import scipy.sparse as sp
# 读取pkl文件
with open('coco_adj.pkl', 'rb') as f:
    adj_matrix = pickle.load(f).astype(np.float32)
    # self.adj_matrix = np.float32(self.adj_matrix)

    # 假设已经创建好了coo_matrix类型的矩阵coo_matrix
    coo_matrix = sp.coo_matrix(adj_matrix)  # 转成coo格式
    i = torch.LongTensor([coo_matrix.row, coo_matrix.col])  # 索引
    v = torch.FloatTensor(coo_matrix.data)  # 数据
    shape = coo_matrix.shape  # 形状
    tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()  # 创建稀疏张量
    adj_matrix = nn.Parameter(tensor, requires_grad=False)

print(adj_matrix.data.max(), adj_matrix.data.min())