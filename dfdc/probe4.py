import torch, pathlib, sys
wp = pathlib.Path('./weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36')
ckpt = torch.load(str(wp), map_location='cpu', weights_only=False)
state = ckpt['state_dict']
state = {k.replace('module.','',1): v for k,v in state.items()}
all_keys = sorted(state.keys())
with open('keys_out.txt','w') as f:
    f.write(f'Total keys: {len(all_keys)}\n')
    f.write('Last 20 keys:\n')
    for k in all_keys[-20:]:
        f.write(f'  {k}  {state[k].shape}\n')
    f.write('First 5 keys:\n')
    for k in all_keys[:5]:
        f.write(f'  {k}  {state[k].shape}\n')
print('Done')
