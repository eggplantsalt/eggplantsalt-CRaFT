from huggingface_hub import hf_hub_download
import json
import pprint

config_path = hf_hub_download('lerobot/pi0fast-base', 'config.json')
config = json.load(open(config_path))

print('=== Pi0Fast 预训练模型期望的图像特征 ===')
image_features = {k: v for k, v in config['input_features'].items() if 'image' in k.lower()}
pprint.pprint(image_features)

print('\n=== 所有输入特征 ===')
pprint.pprint(config['input_features'])

