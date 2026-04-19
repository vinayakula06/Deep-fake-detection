import torch, pathlib
wp = pathlib.Path('./weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36')
ckpt = torch.load(str(wp), map_location='cpu', weights_only=False)
state = ckpt['state_dict']
state = {k.replace('module.','',1): v for k,v in state.items()}
# Print ALL keys
all_keys = sorted(state.keys())
print(f'Total keys: {len(all_keys)}')
print('Last 15 keys (likely head):')
for k in all_keys[-15:]:
    print(f'  {k}  {state[k].shape}')
print('First 5 keys:')
for k in all_keys[:5]:
    print(f'  {k}  {state[k].shape}')
