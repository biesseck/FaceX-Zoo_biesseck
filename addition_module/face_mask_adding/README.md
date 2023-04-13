# COMPILE "render.pyx" (Bernardo)
```
cd FaceX-Zoo_biesseck/addition_module/face_mask_adding/FMA-3D/utils/cython
python setup.py build_ext -i
```


# PIPELINE TO SYNTHETICALLY ADD MASK TO ONE IMAGE FACE (Bernardo)

### 1) Detect face and landmarks
```
cd FaceX-Zoo_biesseck/face_sdk
python api_usage/face_detect.py
python api_usage/face_alignment.py
```

### 2) Add mask
```
cd FaceX-Zoo_biesseck/addition_module/face_mask_adding/FMA-3D
python add_mask_one.py
```
