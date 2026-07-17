# Legacy ResNet-101 Descriptor Audit

## Source

The audit is based on:

`LoopClosure/KITTI/Group5/KITTI_05_resnet.ipynb`

## Preprocessing

Each KITTI image is loaded as RGB and resized to 256 x 256 pixels.

Two crops are extracted:

- right: top=4, left=124, height=124, width=124;
- left: top=4, left=4, height=124, width=124.

Each crop is normalized with:

- mean: (0.36, 0.36, 0.36);
- standard deviation: (0.28, 0.28, 0.28).

## Backbone and representation

The notebook uses ResNet-101 and forwards each crop through:

- conv1;
- bn1;
- relu;
- maxpool;
- layer1;
- layer2;
- layer3;
- layer4;
- avgpool.

The fully connected classification layer is not used.

Each crop produces a tensor with shape:

`[1, 2048, 1, 1]`

The right and left outputs are concatenated along the channel dimension,
in that order:

`right || left`

The resulting legacy image descriptor has shape:

`[1, 4096, 1, 1]`

or 4096 values after flattening.

## Legacy matching semantics

The notebook computes:

```python
distance = torch.cdist(cam_img, db_img, p=2)
max_dist = torch.max(distance)
q
