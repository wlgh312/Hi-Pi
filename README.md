# Hi-Pi

## 프로젝트 목적
요즈음 많은 사람들이 반려묘를 키우는 추세이다. 
하지만 대부분 학교에 다니거나 일을 하기 때문에 반려묘가 혼자 집에 있는 시간이 길다. 때문에 사람들은 반려묘가 혼자 집에 있는 동안 잘 지내는지 알 수 없다. 
또한 같이 있을 때 고양이의 상태가 좋지 않아서 생활 패턴에 변화가 있더라도 잘 알아채지 못한다. 따라서 이번 프로젝트를 통해 고양이의 현 상태를 확인하기로 하였다.
본 프로젝트에서 인공지능을 이용하여 분석하고자 하는 고양이의 상태는 고양이의 주요 행동인 밥 먹는 행동, 물 마시는 행동, 화장실 가는 행동으로 정의하였다.
요약하면, 고양이의 주요 행동들로부터 발생되는 소리를 분석하여 반려묘의 현 상황 및 상태를 확인하는 것이 이번 프로젝트의 최종 목적이다.

## <u>Data</u>
고양이의 주요 행동인 밥 먹기, 물 마시기, 화장실 가기를 확인 대상으로 정하고 다른 이상한 소리가 검출되지 않도록 잡음 또한 분류 대상 중 하나로 잡았다. 또한 기존 소리에 noise와 음악을 각각 추가하여 더 많은 데이터를 수집하였다.  
데이터 학습을 위해 모두 2초 단위로 자른 뒤 spectrogram으로 변환하였다. 
## <u>Jetson</u>
실제로 프로젝트를 구현하기 위해서는 병렬처리가 가능한 Convolution 구현이 필요하였으며, 이를 위해 GPU환경에서 작업이 가능한 NVidia의 Jetson Nano 보드를 사용하였다.
## <u>Result</u>
* jetson_case
<br><img height="200" src="https://github.com/wlgh312/Hi-Pi/blob/master/result/case_infront.png"></img><img height="200" src="https://github.com/wlgh312/Hi-Pi/blob/master/result/case_side.png"></img><br>
* acc
<br><img height="300" src="https://github.com/wlgh312/Hi-Pi/blob/master/result/accuracy.png"></img><br>
* demo
<br><img height="200" src="https://github.com/wlgh312/Hi-Pi/blob/master/result/result_img.png"></img>