from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dropout,Dense,Reshape
from keras import utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob , os
import cv2
from sklearn.model_selection import train_test_split

def main():
    #データセットの.npzを読み込む。任意のパスに変更すること。
    #注：データセットはGitHubに上げてません。
    X_train , X_test , X_val , y_train , y_test , y_val = load_dataset("*.npz")
    train(X_train, X_test, X_val, y_train, y_test, y_val, build_CNN())

def build_CNN():
    #モデルのインスタンス化
    model = Sequential()
    #畳み込み層1
    model.add(Conv2D(64,kernel_size=16,activation="relu",input_shape=(50,50,3)))
    model.add(MaxPool2D(2))
    #畳み込み2
    model.add(Conv2D(64,kernel_size=8,activation="relu"))
    model.add(MaxPool2D(2))
    #畳み込み3
    model.add(Conv2D(64,kernel_size=4,activation="relu"))
    model.add(MaxPool2D(2))
    #平面化
    model.add(Flatten())
    #ドロップアウト--過学習を防ぐ
    model.add(Dropout(0.5))
    #全結合層
    model.add(Dense(128,activation="relu"))
    #最終出力は2(=二値分類)
    model.add(Dense(2,activation="softmax"))
    #モデル構成の表示
    model.summary()
    #モデルをJsonとして保存(シリアライズ)
    json_string = model.to_json()
    open("model.json",'w').write(json_string)
    return model

def load_dataset(path):
    #.npzをロード
    files = glob.glob(path)
    print(files)
    dataset = np.load(files[0])
    X = dataset["X"]
    y = dataset["y"]
    for i in range(len(files)-1):
        ds2 = np.load(files[1+i])
        X = np.vstack((X,ds2["X"]))
        y = np.hstack((y,ds2["y"]))
    #サイズを50x50に圧縮する。OpenCVを使う。
    X_resize = []
    for v in X:
        X_resize.append(cv2.resize(v,(50,50)))
    X_resize = np.asarray(X_resize)
    #Kerasはラベルを単位ベクトルとして扱うので変更
    y = utils.np_utils.to_categorical(y,2)
    #データを訓練データ、テストデータ、バリデーションデータに分ける
    #総データ数1000、訓練データ600、テストデータ250、バリティエーションデータ150
    X_tarin , X_test , y_train ,y_test = train_test_split(X_resize,y,test_size=0.4,random_state=19)
    X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,train_size=0.625,random_state=19)
    return  X_tarin , X_test , X_val , y_train ,y_test, y_val

def train(X_train,X_test,X_val,y_train,y_test,y_val,model):
    #画像を加工
    #データセットの標準偏差で画像を正規化。
    #ランダム回転範囲：20°
    #縦横に0.2ランダムシフト
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2)
    datagen.fit(X_train)
    #目的関数にSGD、誤差関数にloglessを用いる。二値分類は'binary_crossentropy'
    model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    #学習
    #EarlyStoppingを用いて過学習を防ぐ。
    es_cb = EarlyStopping()
    model.fit_generator(datagen.flow(X_train,y_train),validation_data=(X_val,y_val),callbacks=[es_cb],epochs=30,steps_per_epoch=300)
    #重りを保存
    model.save_weights("drive/Colab/nn_face_check_sc/model_weights.hdf5")
    #テスト
    r = model.evaluate(X_test,y_test)
    print(r)

if __name__=="__main__" :
    main()