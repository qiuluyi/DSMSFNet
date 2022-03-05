import timm

def get_model(model_name, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
    return model

def get_nsdm_model(model_name, pretrained=True,checkpoint_path=''):
    model = timm.create_model(model_name, pretrained=pretrained,in_chans=1, features_only=True)
    return model