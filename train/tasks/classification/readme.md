# Bonnetal Classification

This readme contains simple usage explanations of how the training, evaluation, test inference, and export to ONNX/TensorRT are performed.

![](https://i.ibb.co/J2ryMKD/im-a-tennis-ball.png)
Example of a PhD student being ignored by an algorithm (and why full image classification is a bad idea). This is from a ResNet50 trained on ImageNet (person is not a class, and neither is PhD student :stuck_out_tongue:).

---
## Train a Network

This part of the framework provides a set of apps to train a full image classification CNN. It uses a backbone from our library of backbones, and attaches global pooling and a linear classifier layer in the end. Training is performed by a weighted cross entropy loss of the softmax-ed logits.

The training parameters are selected through a `cfg.yaml` file containing 

- `train`: Dictionary with training parameters, such as learning rate, momentum, batch size, warmup phase, learn rate decay, weight decay, model averaging, etc.
- `backbone`: Dictionary with backbone selector and configurator, such as dropout in convolutional layers, batch normalization decay, output stride, etc. This part is common to all tasks using the common backbones.
- `head`: Classification head config, such as dropout before the linear layer.
- `dataset`: Name and location of the dataset, sizes of images, means and standard deviations for input normalization, class definition, and class weights (if imbalanced).

#### To train a network

- For help:

  ```sh
  $ ./train.py --help
  ```

- From scratch (if the weights exist and the images are RGB, the backbones download imagenet weights from the internet, so it is not really from scratch):

    ```sh
  $ ./train.py -c /path/to/cfg.yaml -l /path/to/put/log/
  ```

- From a pretrained model:

  ```sh
  $ ./train.py -c /path/to/cfg.yaml -p /path/to/pretrained/ -l /path/to/put/log/
  ```

- From scratch for real (no imagenet backbone, you need to fake it):

  ```sh
  $ ./train.py -c /path/to/cfg.yaml -p /dev/null -l /path/to/put/log/
  ```

#### To evaluate a network

- It uses the same script as the train but with an extra flag:

  ```sh
  $ ./train.py -p /path/to/pretrained/ --eval
  ```

---
## Test Inference

These are some scripts to infer images and videos with the trained CNNs

#### Infer images

- Help:

  ```sh
  $ ./infer_img.py --help
  ```

- With native pytorch implementation:

  ```sh
  $ ./infer_img.py -p /path/to/pretrained/ -i /path/to/images # can be multiple images
  $ ./infer_img.py -p /path/to/pretrained/ -i /path/to/images -b native
  ```
- With a different backend (we have implemented picking up a traced pytorch model, an ONNX with Caffe2, and TensorRT). This is here to try if the homologous backbone will work during C++ inference. The `make_deploy_model.py` script traces the CNN and gives us binaries to pick up with different backbones. Furthermore, when using the `infer_img.py` file, the TensorRT engine that was created from the ONNX model is saved in the inference folder, because it takes a while to create, and you may want to use it later. If you already ran it, and want to change computers, DELETE `model.trt`:
  
  ```sh
  # get ready for inference
  $ ./make_deploy_model -p /path/to/pretrained/ -l /path/to/save/inference/ready
  # infer from inference model
  $ ./infer_img.py -p /path/to/save/inference/ready -i /path/to/images -b pytorch # or
  $ ./infer_img.py -p /path/to/save/inference/ready -i /path/to/images -b caffe2 # or
  $ ./infer_img.py -p /path/to/save/inference/ready -i /path/to/images -b tensorrt
  ```


#### Infer video/webcam

- Help:

  ```sh
  $ ./infer_video.py --help
  ```

- With native pytorch implementation:

  ```sh
  $ ./infer_video.py -p /path/to/pretrained/ -v /path/to/video
  $ ./infer_video.py -p /path/to/pretrained/ -v /path/to/video -b native # same thing
  ```

- Infer Webcam instead of video:

  ```sh
    $ ./infer_video.py -p /path/to/pretrained/ # simply drop the video and it will look for /dev/video0
  ```
  
- With a different backend (we have implemented picking up a traced pytorch model, an ONNX with Caffe2, and TensorRT). This is here to try if the homologous backbone will work during C++ inference. The `make_deploy_model.py` script traces the CNN and gives us binaries to pick up with different backbones. Furthermore, when using the `infer_img.py` file, the TensorRT engine that was created from the ONNX model is saved in the inference folder, because it takes a while to create, and you may want to use it later. If you already ran it, and want to change computers, DELETE `model.trt`:
  
  ```sh
  # get ready for inference
  $ ./make_deploy_model -p /path/to/pretrained/ -l /path/to/save/inference/ready
  # infer from inference model
  $ ./infer_video.py -p /path/to/save/inference/ready -v /path/to/video -b pytorch # or
  $ ./infer_video.py -p /path/to/save/inference/ready -v /path/to/video -b caffe2 # or
  $ ./infer_video.py -p /path/to/save/inference/ready -v /path/to/video -b tensorrt
  ```

## Make inference model

- Get ready for C++ deployment:

  ```sh
  $ ./make_deploy_model -p /path/to/pretrained/ -l /path/to/save/inference/ready
  ```
  This generates a Pytorch model and an ONNX model. The TensorRT one gets created and profiled for efficiency from the ONNX during the first run, either here or in the C++ interface. Once built, the interfaces save it as an engine that get's deserialized the next execution, so if you change computers, delete this file, as it will not be the proper format/optimal.

- Change the size of the image for inference, if you want to infer on difference size than what the network was trained on (if scale of objects didn't change, it shouldn't be a problem for accuracy). Because this script saves the modified config file with the new size, the config dictionaries will be scrambled:

  ```sh
  $ ./make_deploy_model -p /path/to/pretrained/ -l /path/to/save/inference/ready --new_w XXX --new_h YYY
  ```