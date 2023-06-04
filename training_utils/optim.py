import torch
import torch.nn as nn


def get_optimizer(model, config):
    lr = config['lr']
    lr_scaling_backbone = config['lr_scaling_backbone']
    train_bb = config['train_backbone']

    if hasattr(model, "backbone"):
        encoder_string = "model.backbone"
    elif hasattr(model, "encoder"):
        encoder_string = "model.encoder"
    else:
        raise Exception()
    
    if hasattr(model, "classifier"):
        decoder_string = "model.classifier"
    elif hasattr(model, "decoder"):
        decoder_string = "model.decoder"
    else:
        raise Exception()
    
    if train_bb:
        if config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD([{'params': eval(encoder_string).parameters()},
                                         {'params': eval(decoder_string).parameters(), 'lr': lr}],
                                        lr=lr*lr_scaling_backbone,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=config['weight_decay']
                                        )
        
        if config['optimizer'] == 'rms':
            optimizer = torch.optim.RMSprop([{'params': eval(encoder_string).parameters()},
                                             {'params': eval(decoder_string).parameters(), 'lr': lr}],
                                            lr=lr*lr_scaling_backbone,
                                            momentum=0.9,
                                            weight_decay=config['weight_decay']
                                            )
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam([{'params': eval(encoder_string).parameters()},
                                          {'params': eval(decoder_string).parameters(), 'lr': lr}],
                                         lr=lr*lr_scaling_backbone
                                         )
    else:
        for p in eval(encoder_string).parameters():
            p.requires_grad = False
        if config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(eval(decoder_string).parameters(),
                                        lr=lr,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=config['weight_decay']
                                        )
        
        if config['optimizer'] == 'rms':
            optimizer = torch.optim.RMSprop(eval(decoder_string).parameters(),
                                             lr=lr,
                                             momentum=0.9,
                                             weight_decay=config['weight_decay']
                                            )
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(eval(decoder_string).parameters(),
                                          lr=lr,
                                         )
    return optimizer