import torch
premodel = torch.load('D:\project_python\SparseInst\pretrained_models/sparse_inst_r50vd_dcn_giam_aug_67dc06.pth')

aftermodel = torch.load('D:\project_python\SparseInst\pretrained_models/3d_layer1.pth')

# premodel['model']['backbone.bottom_up.stem.conv1.weight'] = aftermodel['model']['backbone.bottom_up.stem.conv1.weight']

torch.save(aftermodel, 'D:\project_python\SparseInst\pretrained_models/test.pth')
print("model saved")