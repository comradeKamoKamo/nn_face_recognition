import numpy as np
import os, glob, random,cv2

#画像データをnumpy配列データに変換する。
#注意：このコードは参照先のデータがないので機能しません。

#乱数の種を固定
random.seed(19)
np.random.seed(19)

def main():
    #顔のラベルは0
    image_to_db(250,500,label=0,path_in="images/lfw/lfw/*/*.jpg",path_out="dataset_lfw.npz")
    #実験室のラベルは1
    image_to_db(250,500,label=1,path_in="images/bio/*.png",path_out="dataset_bio.npz")

def image_to_db(photo_size,photo_count,label,path_in,path_out,first_index=0):
    X = []              #データ
    y = []              #ラベル

    files = glob.glob(path_in)
    random.shuffle(files)

    for i in range(first_index,first_index+photo_count):
        print(files[i])
        img = cv2.imread(files[i])

        #正規化
        data = np.asarray(img)
        data = data / 256
        data.reshape(photo_size,photo_size,3)

        X.append(data)
        y.append(label)

    X = np.array(X,dtype=float)
    np.savez(path_out,X=X,y=y)
    print("saved!")

if __name__=="__main__":
    main()

