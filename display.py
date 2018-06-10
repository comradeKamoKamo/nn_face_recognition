from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import glob , os, cv2, time

def main():
    #モデルをロード
    model = load_model("model.json","model_weights.hdf5")
    #目的関数にSGD、誤差関数にloglessを用いる
    model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    #映像入力0番からインスタンス化
    cap = cv2.VideoCapture(0)
    i = 0
    try:
       while(True):
            #Webカメラから映像を取得
            ret, frame = cap.read()
            if ret:
                #画像をトリミング
                dst_f = frame[0:480,80:560]
                #リサイズ
                img = cv2.resize(dst_f,(50,50))
                #numpy配列へ
                data = np.asarray(img,dtype=float)
                #正規化
                data = data / 256
                data = data.reshape(1,50,50,3)
                #予測
                r = model.predict(data,verbose=0)[0]                    
                if r[0]>=0.4:
                    print(pycolor.RED+"This may be a FACE."+pycolor.END,r)
                    cv2.putText(dst_f,"FACE!",(0,400),cv2.FONT_HERSHEY_DUPLEX,1.3,(0,0,255))
                else:
                    print(pycolor.BLUE+"This may NOT be a face.",pycolor.END,r)
                
                #表示
                cv2.imshow('Face Recognition', dst_f)
                cv2.waitKey(20)
            else:
                time.sleep(2)
    except KeyboardInterrupt:
        print("Interrupt!")
        cap.release()
        cv2.destroyAllWindows()
        exit()

def load_model(model_json_path,model_weights_path):
    #jsonからモデル構造をロード（デシリアライズ）
    json_string = open(model_json_path,'r').read()
    model = model_from_json(json_string)
    #モデルの重みをロード
    model.load_weights(model_weights_path)
    return model

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'

if __name__=="__main__":
    main()


