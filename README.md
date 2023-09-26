# Airway Segmentation And Generation Labeling Tool

[**NaviAirway: a Bronchiole-sensitive Deep Learning-based Airway Segmentation Pipeline**](https://arxiv.org/abs/2203.04294) 논문의 모델을 기반으로 airway segmentation을 얻은 후, 얻어진 segmentation에서 각 voxel이 어떤 generation에 속하는지 labeling합니다.

|<img src="./figs/segmentation.png" width=300>|<img src="./figs/generation_labeling.png" width=300>|
|:---:|:---:|
|Airway Segmentation|Generation Labeling된 Airway Segmentation|

## 실행 방법
현재로서는 python 및 conda 환경을 구축한 다음에 코드를 실행할 수 있습니다. 이런 환경 설정 없이 stand-alone 방식으로 실행될 수 있는 패키지를 만드는 방법은 현재 구상 중입니다.

### 1. Anaconda 가상환경 생성 및 활성화

```
conda create -p ./.conda
conda activate -p ./.conda
```

### 2. 필요 패키지 설치
```
pip install -r requirements.txt
```
PyTorch의 경우, GPU를 이용한 추론을 위해서는 추가적인 설정이 필요할 수 있습니다. [여기](https://pytorch.org/get-started/locally/)를 참조해 주세요.

### 3. infer.py 실행
CT 이미지 파일에서 airway segmentation 결과를 
```
# conda 가상환경이 활성화되지 않았다면 아래 줄의 주석을 지우고 실행
# conda activate -p ./.conda
python infer.py \
    --weight_path WEIGHT_PATH [WEIGHT_PATH ...] \
    --image_path IMAGE_PATH [IMAGE_PATH ...] \
    [--select_dir] \
    --save_path SAVE_PATH \
    [--threshold THRESHOLD] \
    [--segmentation_only] \
    [--branch_penalty BRANCH_PENALTY] \
    [--prune_threshold PRUNE_THRESHOLD] \
    [--use_bfs] \
    [--do_not_add_broken_parts] \
    [--device DEVICE]
```
대괄호 안에 있는 매개변수는 명령어에 포함되지 않아도 되는 매개변수입니다. <br>
각 매개변수에 대한 설명은 다음과 같습니다.

* ```--weight_path```: 추론에 사용할 모델 가중치의 경로. 2개 이상의 가중치를 사용하여 앙상블 추론도 가능합니다.
* ```--image_path```: 추론할 CT 이미지의 파일(또는 파일이 들어있는 폴더) 경로. 추론에 사용할 이미지는 [ITK-SNAP](http://www.itksnap.org/) 등의 툴을 이용해 ```*.nii.gz``` 형식의 파일로 변환된 상태여야 합니다.
* ```--select_dir```: 이 매개변수가 명령어에 포함되어 있으면, ```--image_path``` 매개변수의 동작이 달라집니다.<br>
```--select_dir```가 명령어에 포함되어 있지 않으면, ```---image_path```의 하위 매개변수인 각 ```IMAGE_PATH```는 CT 이미지 **파일**의 경로여야 합니다. 이때 프로그램은 각 ```IMAGE_PATH```에 해당되는 이미지 파일에 대해 추론을 진행합니다.<br>
```--select_dir```가 명령어에 포함되어 있으면, ```---image_path```의 하위 매개변수인 각 ```IMAGE_PATH```는 CT 이미지 파일이 있는 **폴더**의 경로여야 합니다. 이때 프로그램은 각 ```IMAGE_PATH```에 해당되는 폴더 내에 있는 CT 이미지 파일들에 대해 추론을 진행합니다.<br>
* ```--save_path```: 결과 파일들을 저장할 **폴더**의 경로입니다. 존재하지 않는 폴더라면 프로그램이 실행될 때 폴더를 알아서 만들어 줍니다.
* ```--threshold```: 이미지를 segmentation 모델에 넣어서 출력되는 값은 일차적으로 각 voxel이 airway 분류에 속할 확률값입니다. 이 확률값이 ```THRESHOLD```의 값 이상이면 해당 voxel을 airway로 판단하고, 그렇지 않으면 airway가 아니라고 판단하게 됩니다. 기본값은 ```0.7```입니다.
* ```--segmentation_only```: 이 매개변수가 명령어에 포함되어 있으면, generation labeling 및 관련 작업들을 수행하지 않고 segmentation 결과 및 관련 정보만 파일로 저장합니다.
* ```--branch_penalty```, ```--prune_threshold```, ```--use_bfs```, ```--do_not_add_broken_parts```: 알고리즘 비교/분석 및 디버깅을 위해 만들어 둔 매개변수입니다. 웬만하면 쓰지 마세요.
* ```--device```: 딥러닝 모델을 돌릴 연산 장치를 설정합니다. 기본값은 ```cuda```로 설정되어 있어 nVidia GPU로 연산을 하게 되며, nVidia GPU가 없거나 사용 중 오류가 발생하는 경우 ```cpu```로 값을 변경하여 CPU로 연산을 하게 하면 문제가 해결될 수 있습니다.



### 4. break_by_gen.py 실행
```infer.py``` 또는 외부 segmentation 모델로 airway segmentation만 얻었을 때, 이 segmentation에서 generation labeling만 실행합니다. <br>
아직 README를 제대로 작성하고 일반 사용자가 쉽게 실행할 수 있을 정도로 코드를 다듬지 못했습니다. 나중에 만들어둘게요.


## 출력 파일 설명
결과 파일이 들어있는 폴더는 다음과 같은 구조입니다.
```
<save_path>
|
|-- by_gen
|   |-- case1_segmentation_by_gen.nii.gz
|   |-- case1_segmentation_by_gen_left.nii.gz
|   |-- case1_segmentation_by_gen_right.nii.gz
|   |-- case2_segmentation_by_gen.nii.gz
|   |-- case2_segmentation_by_gen_left.nii.gz
|   |-- case2_segmentation_by_gen_right.nii.gz
|   |-- ....
|
|-- centerline
|   |-- case1_segmentation_by_gen.nii.gz
|   |-- case2_segmentation_by_gen.nii.gz
|   |-- ...
|
|-- extended_segment
|   |-- case1_segmentation.nii.gz
|   |-- case2_segmentation.nii.gz
|   |-- ...
|
|-- extended_segment_before_preprocess
|   |-- case1_segmentation.nii.gz
|   |-- case2_segmentation.nii.gz
|   |-- ...
|
|-- orig_segment
|   |-- case1_segmentation.nii.gz
|   |-- case2_segmentation.nii.gz
|   |-- ...
|
|-- orig_segment_before_preprocess
|   |-- case1_segmentation.nii.gz
|   |-- case2_segmentation.nii.gz
|   |-- ...
|
|-- trace_slice_area_info
|   |-- trace_slice_area_info_case1.csv
|   |-- trace_slice_area_info_case2.csv
|   |-- ...
|
|-- generation_info.csv
|-- pixdim_info.csv
|-- trace_volume_by_gen_info.csv
```

