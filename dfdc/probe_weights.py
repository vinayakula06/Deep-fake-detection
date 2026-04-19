import torch, pathlib, sys
wp = pathlib.Path('./weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36')
print("Loading...", flush=True)
try:
    state = torch.load(str(wp), map_location='cpu', weights_only=False)
    print('Type:', type(state))
    if isinstance(state, dict):
        print('Top-level keys:', list(state.keys())[:10])
        for k in list(state.keys())[:3]:
            v = state[k]
            print(f'  {k}: {type(v).__name__}', getattr(v,'shape',''))
    else:
        print('Repr:', repr(state)[:300])
except Exception as e:
    print('ERROR:', e)
    import traceback; traceback.print_exc()
