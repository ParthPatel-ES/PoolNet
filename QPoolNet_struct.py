PoolNet(
  (base): ResNet_locate(
    (resnet): QuantizableResNet(
      (conv1): QuantizedConvReLU2d(3, 64, kernel_size=(7, 7), stride=(2, 2), scale=1.0, zero_point=0, padding=(3, 3))
      (bn1): Identity()
      (relu): Identity()
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)
      (layer1): Sequential(
        (0): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(64, 64, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (downsample): Sequential(
            (0): QuantizedConv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
            (1): Identity()
          )
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (1): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(256, 64, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (2): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(256, 64, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
      )
      (layer2): Sequential(
        (0): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(256, 128, kernel_size=(1, 1), stride=(2, 2), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (downsample): Sequential(
            (0): QuantizedConv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), scale=1.0, zero_point=0)
            (1): Identity()
          )
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (1): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(512, 128, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (2): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(512, 128, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (3): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(512, 128, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
      )
      (layer3): Sequential(
        (0): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(512, 256, kernel_size=(1, 1), stride=(2, 2), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (downsample): Sequential(
            (0): QuantizedConv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), scale=1.0, zero_point=0)
            (1): Identity()
          )
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (1): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (2): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (3): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (4): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (5): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(1, 1))
          (bn2): Identity()
          (conv3): QuantizedConv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
      )
      (layer4): Sequential(
        (0): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(512, 512, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(2, 2), dilation=(2, 2))
          (bn2): Identity()
          (conv3): QuantizedConv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (downsample): Sequential(
            (0): QuantizedConv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
            (1): Identity()
          )
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (1): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(512, 512, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(2, 2), dilation=(2, 2))
          (bn2): Identity()
          (conv3): QuantizedConv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
        (2): QuantizableBottleneck(
          (conv1): QuantizedConvReLU2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn1): Identity()
          (conv2): QuantizedConvReLU2d(512, 512, kernel_size=(3, 3), stride=(1, 1), scale=1.0, zero_point=0, padding=(2, 2), dilation=(2, 2))
          (bn2): Identity()
          (conv3): QuantizedConv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
          (bn3): Identity()
          (relu): QuantizedReLU()
          (skip_add_relu): QFunctional(
            scale=1.0, zero_point=0
            (activation_post_process): Identity()
          )
          (relu1): Identity()
          (relu2): Identity()
        )
      )
      (quant): Quantize(scale=tensor([0.0079]), zero_point=tensor([0]), dtype=torch.quint8)
      (dequant): DeQuantize()
    )
    (ppms_pre): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (ppms): ModuleList(
      (0): Sequential(
        (0): AdaptiveAvgPool2d(output_size=1)
        (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): AdaptiveAvgPool2d(output_size=3)
        (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): AdaptiveAvgPool2d(output_size=5)
        (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): ReLU(inplace=True)
      )
    )
    (ppm_cat): Sequential(
      (0): Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
    )
    (infos): ModuleList(
      (0): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
    )
  )
  (deep_pool): ModuleList(
    (0): DeepPoolLayer(
      (pools): ModuleList(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (2): AvgPool2d(kernel_size=8, stride=8, padding=0)
      )
      (convs): ModuleList(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (relu): ReLU()
      (conv_sum): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (conv_sum_c): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
    (1): DeepPoolLayer(
      (pools): ModuleList(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (2): AvgPool2d(kernel_size=8, stride=8, padding=0)
      )
      (convs): ModuleList(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (relu): ReLU()
      (conv_sum): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (conv_sum_c): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
    (2): DeepPoolLayer(
      (pools): ModuleList(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (2): AvgPool2d(kernel_size=8, stride=8, padding=0)
      )
      (convs): ModuleList(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (relu): ReLU()
      (conv_sum): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (conv_sum_c): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
    (3): DeepPoolLayer(
      (pools): ModuleList(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (2): AvgPool2d(kernel_size=8, stride=8, padding=0)
      )
      (convs): ModuleList(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (relu): ReLU()
      (conv_sum): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (conv_sum_c): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
    (4): DeepPoolLayer(
      (pools): ModuleList(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (2): AvgPool2d(kernel_size=8, stride=8, padding=0)
      )
      (convs): ModuleList(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (relu): ReLU()
      (conv_sum): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
  )
  (score): ScoreLayer(
    (score): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
  )
  (convert): ConvertLayer(
    (convert0): ModuleList(
      (0): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
      (4): Sequential(
        (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
    )
  )
)