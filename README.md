# nn_face_recognition
CNNを用いた機械学習で人の顔を認識するプログラムを作る。

# About
　私はこの時点で日本の高校生です。初学者なのでいろいろとメチャクチャです。  
　これは、2018年度文化祭で部活動で展示する発表内容です。Keras+TensorFlowを使って、畳み込みニュートラルネットワークを構成し、人の顔を機械学習によって認識させることを目的とします。  
　display.pyが実物展示用のプログラム。train.pyが最終的な学習コード。image_to_db.pyは画像データをnumpyの配列に変換して保存するコードです。
 データセットとして、部活の活動場所たる生物実験室の画像500枚と、LFWの顔データセットからランダムに画像500枚を利用し、二値分類を行います。データはここに上げていません。学習済みのモデルの重みデータはmodel_weights.hdf5です。  
>LFW顔データセット→http://vis-www.cs.umass.edu/lfw/  
# Environment
## Anacondaの環境
- Python 3.5.4
- TensorFlow 1.0.1
- Keras 2.0.5
- OpenCV, scikit-learn, numpy
## PC
- Microsoft Windows 10 Home / 10.0.17134 ビルド 17134
- Intel Celeron CPU N3350  
- VGA UVC WebCamera    
悲しいかなGPUもない低スペ...そのため、学習などにはGoogle Colabも用いた。  
display.pyについては0番の映像入力がVGA画質(640x480)である前提で書かれています。
  
# Thank you for reading.
　読んでくれてありがとうございます。

