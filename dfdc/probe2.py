import torch, pathlib
wp = pathlib.Path('./weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36')
ckpt = torch.load(str(wp), map_location='cpu', weights_only=False)
state = ckpt['state_dict']
state = {k.replace('module.','',1): v for k,v in state.items()}
fc_keys = [(k,v.shape) for k,v in state.items() if k.startswith('fc')]
enc_cls = [(k,v.shape) for k,v in state.items() if 'encoder.classifier' in k]
print('Keys starting with fc:')
for k,s in fc_keys: print(f'  {k}  {s}')
print('encoder.classifier keys:')
for k,s in enc_cls: print(f'  {k}  {s}')
