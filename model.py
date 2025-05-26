from torchvision import models, transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
def get_model(model_name='resnet18', pretrained=True, num_classes=101):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in list(model.parameters())[:-2]: 
            param.requires_grad = False

def configure_model_for_finetuning(model, feature_extract, lr_backbone, lr_fc, momentum, weight_decay):
    set_parameter_requires_grad(model, feature_extract)
    if feature_extract:
        params_to_update = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.SGD(params_to_update, lr=lr_fc, momentum=momentum, weight_decay=weight_decay)
    else:
        params_to_update = [
            {'params': [param for name, param in model.named_parameters() 
                       if not ('fc' in name or 'classifier' in name)], 'lr': lr_backbone},
            {'params': [param for name, param in model.named_parameters() 
                       if 'fc' in name or 'classifier' in name], 'lr': lr_fc}
        ]
        
        optimizer = optim.SGD(params_to_update, momentum=momentum, weight_decay=weight_decay)
    
    return optimizer

