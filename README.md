# 👀Early_Vision_Project
Computer Vision 분야에서 Deep Learning은 엄청난 성공을 거뒀습니다.  
ImageNet Competition에서 CNN을 사용한 AlexNet이 우승을 하고 기존 ML기반에서 DL로 연구 패러다임이 넘어가는 계기가 됐습니다.  
  
Deep Learning은 Vision 분야의 기본적인 Task에서 뛰어난 성과를 보이지만 데이터를 이해하고 처리하는 초기단계에서는 여전히 Early Vision 기술의 이해가 필수적입니다. 뿐만아니라 CNN Model이 low-level feature로 부터 high-level feature까지 점진적으로 Layer를 통과하며 이미지를 인식하는 과정 또한 Early Vision기술을 알고있어야 깊이 이해할 수 있습니다.
  
Early Vision의 이해 없이 딥러닝 구조만 이용하는 것은 데이터에 내재된 의미를 완전히 이해할 수 없을 것 입니다. 따라서 이 레포에서는 Early Vision을 공부한 내용을 공유하려고 합니다.

## 💻Image Filter

Salt & Pepper Noise, Gaussian noise등이 포함된 이미지에서 noise를 제거하기 위한 Filter입니다. CNN과 같이 특정한 weight를 가진 filter가 Sliding Window를 하며 noise를 제거합니다. noise 제거를 위한 Mean,Median,Gaussian Filter와 Image Enhancing을 위한 High-Boost Filter를 구현했고 해당 폴더 내의 Readme에 자세한 내용을 기술했습니다. 
  
### Original Image
![image](https://github.com/Pulyong/Early_Vision_Project/assets/76218918/e4d814e0-e403-4f22-bc89-9937a6d0bd6f)

### 5x5 median Filter
![image](https://github.com/Pulyong/Early_Vision_Project/assets/76218918/b2a26bdd-05b6-43ce-87a8-d81cb204520b)

## ❓Image Classification
 <a href='https://github.com/google/recaptcha' target='_blank'>Recaptcha</a> dataset의 12개 class중 Mountain과 Other class를 제외한 10개의 class를 Deep Learning을 사용하지 않고 classification하는 Task를 수행하는 모델을 구현했습니다. SIFT,LBP를 사용했고, K-means를 이용한 Bag of visual words 방식으로 clustering하여 KNN으로 classification을 수행했습니다. classification뿐만 아니라 KNN 모델 상의 거리가 가장 가까운 10개의 image를 찾는 retrieval Task도 수행했습니다. 자세한 내용은 폴더 내의 Readme에 있습니다.
   
![image.jpg1](https://github.com/Pulyong/Early_Vision_Project/assets/76218918/dfb25d25-a15a-47e0-a684-b83ce591d348) |![image.jpg2](https://github.com/Pulyong/Early_Vision_Project/assets/76218918/6bc0aadd-fb7d-4bde-ac04-88f843c4498a)
--- | --- | 

## 📼Video Detection
Image가 아닌 Video의 Object Detection을 색상 채널 변경이나 Morphology연산을 이용하여 수행했습니다. Threshold를 정해서 Connected Component알고리즘으로 bounding box를 만들어 Object를 표시합니다. Hand Detection, Car Detection을 수행했습니다.

### Hand Detection
![output (online-video-cutter com) (1)](https://github.com/Pulyong/Early_Vision_Project/assets/76218918/c4195ea1-2335-42ef-a88c-4b12cf32f7fb)

### Car Detection
![output_car (online-video-cutter com)](https://github.com/Pulyong/Early_Vision_Project/assets/76218918/60b9add7-9674-402b-b7e7-f148a7e801f2)
