#!/bin/bash
# FlexASR Linear with EfficientNet
python3 validate_compilation.py efficientnet --configs im2col linear-rewrites --get-stats
# HLSCNN on EfficientNet
python3 validate_compilation.py efficientnet --configs hlscnn-conv2d --get-stats
# VTA on EfficientNet
python3 validate_compilation.py efficientnet --configs im2col vta-dense --get-stats

# MobileNet on FlexASR
python3 validate_compilation.py mobilenetv2 --configs im2col linear-rewrites --get-stats 
# MobileNet on HLSCNN
python3 validate_compilation.py mobilenetv2 --configs hlscnn-conv2d --get-stats
# MobileNet on VTA
python3 validate_compilation.py mobilenetv2 --configs im2col vta-dense --get-stats

# ResMLP on FlexASR
python3 validate_compilation.py resmlp --configs im2col linear-rewrites --get-stats
# ResMLP on VTA
python3 validate_compilation.py resmlp --configs im2col vta-dense --get-stats

# ResNet20 on FlexASR
python3 validate_compilation.py resnet20 --configs im2col linear-rewrites --get-stats 
# ResNet20 on HLSCNN
python3 validate_compilation.py resnet20 --configs hlscnn-conv2d --get-stats
# ResNet20 on VTA
python3 validate_compilation.py resnet20 --configs im2col vta-dense --get-stats

# Q-MobileNet on VTA (quantize after matching)
python3 validate_compilation.py mobilenetv2 --configs im2col vta-dense --get-stats

# LSTM on FlexASR
python3 validate_compilation.py lstm-for-pldi --configs flexasr-lstm --get-stats
# LSTM on HLSCNN
python3 validate_compilation.py lstm-for-pldi --configs hlscnn-conv2d --get-stats
# LSTM on VTA
python3 validate_compilation.py lstm-for-pldi --configs vta-dense --get-stats  

# Transformer on FlexASR
python3 validate_compilation.py transformer --configs flexasr-lstm linear-rewrites --get-stats
# Transformer on HLSCNN
python3 validate_compilation.py transformer --configs hlscnn-conv2d --get-stats
# Transformer on VTA
python3 validate_compilation.py transformer --configs vta-dense --get-stats

# Resnet50 from tensorflow
python3 validate_compilation.py resnet50_simplifyinference_from_tf --configs flexasr-lstm linear-rewrites --get-stats
python3 validate_compilation.py resnet50_simplifyinference_from_tf --configs hlscnn-conv2d --get-stats
python3 validate_compilation.py resnet50_simplifyinference_from_tf --configs im2col vta-dense --get-stats

# Resnet50 from pytorch
python3 validate_compilation.py resnet50_simplifyinference_from_pytorch --configs flexasr-lstm im2col linear-rewrites --get-stats
python3 validate_compilation.py resnet50_simplifyinference_from_pytorch --configs hlscnn-conv2d --get-stats
python3 validate_compilation.py resnet50_simplifyinference_from_pytorch --configs im2col vta-dense --get-stats

# Resnet50 from onnx
python3 validate_compilation.py resnet50_simplifyinference_from_onnx --configs flexasr-lstm im2col linear-rewrites --get-stats
python3 validate_compilation.py resnet50_simplifyinference_from_onnx --configs hlscnn-conv2d --get-stats
python3 validate_compilation.py resnet50_simplifyinference_from_onnx --configs im2col vta-dense --get-stats