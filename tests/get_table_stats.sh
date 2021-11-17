#!/bin/bash
# FlexASR Linear with EfficientNet
python3 validate_compilation.py efficientnet --configs linear-rewrites --get-stats
# HLSCNN on EfficientNet
python3 validate_compilation.py efficientnet --configs hlscnn-conv2d --get-stats
# VTA on EfficientNet
python3 validate_compilation.py efficientnet --configs im2col vta-dense --get-stats

# MobileNet on FlexASR
python3 validate_compilation.py mobilenet --configs linear-rewrites --get-stats 
# MobileNet on HLSCNN
python3 validate_compilation.py mobilenet --configs hlscnn-conv2d --get-stats
# MobileNet on VTA
python3 validate_compilation.py mobilenet --configs im2col vta-dense --get-stats

# ResMLP on FlexASR
python3 validate_compilation.py resmlp --configs linear-rewrites --get-stats
# ResMLP on VTA
python3 validate_compilation.py resmlp --configs im2col vta-dense --get-stats

# ResNet20 on FlexASR
python3 validate_compilation.py resnet20 --configs linear-rewrites --get-stats 
# ResNet20 on HLSCNN
python3 validate_compilation.py resnet20 --configs hlscnn-conv2d --get-stats
# ResNet20 on VTA
python3 validate_compilation.py resnet20 --configs im2col vta-dense --get-stats

# Q-MobileNet on VTA (quantize after matching)
python3 validate_compilation.py mobilenetv2 --configs im2col vta-dense --get-stats