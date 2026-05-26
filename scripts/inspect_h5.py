import h5py
import json

weights_path = '/workspace/models/attention_unet/Trained models/retina_attentionUnet_150epochs.hdf5'

with h5py.File(weights_path, 'r') as f:
    if 'model_config' in f.attrs:
        config = json.loads(f.attrs['model_config'].decode('utf-8'))
        
        # Let's print all layers of type 'Lambda' and their config
        layers = config['config']['layers']
        for layer in layers:
            if layer['class_name'] == 'Lambda':
                print(f"Layer Name: {layer['name']}")
                print(f"Config: {layer['config']}")
                print("-" * 50)
    else:
        print("model_config not found in attributes.")
