# Ultra-lightweight-face-detection
Easy to use ultra-lightweight-face-detection based on the OpenVINO public model zoo


# Once you have prepared your python environment and installed openvino

Download the pre-trained model using openvino's downloader tool as following:

```bash
$ omz_downloader --name ultra-lightweight-face-detection-rfb-320 --output_dir model
```

Convert the downloaded model to OpenVINO's Intermediate Representation by using:

```
$ omz_converter --name ultra-lightweight-face-detection-rfb-320 --download_dir model --output_dir model --precision=FP16
```

And done! now you can use the "FaceDetector" class (look at src/face_detector.py), which make very easy to use that model with Openvinos Python API.
you just need to specify the `.xml` file when instanciating the face detector object.