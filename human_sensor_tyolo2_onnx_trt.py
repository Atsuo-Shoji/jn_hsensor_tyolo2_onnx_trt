#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#汎用
import cv2
import numpy as np
import sys
import os
import traceback
import argparse

#NVIDIA提供
from get_engine import get_engine
import common
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import RPi.GPIO as GPIO

#自作
from Yolov2_bboxes_generator import Yolov2_bboxes_generator


GST_STR_CSI = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1, sensor-id=%d \
    ! nvvidconv flip-method=%d ! video/x-raw, width=%d, height=%d, format=BGRx \
    ! videoconvert \
    ! appsink' #cv2へのCSIカメラからのキャプチャ指示のGStreamerコマンド文字列

FPS = 30 #キャプチャーのフレームレート
WINDOW_NAME = 'Human Sensor'#キャプチャー映像を出力するウィンドウの名前
YOLO2_NN_INPUT_SIZE = (416, 416) #YOLO v2のNNへの入力画像のサイズ
ONNX_FILE_PATH = './tyolov2NN_trained/tyolov2NN_trained.onnx'
TRT_ENGINE_FILE_PATH = './tyolov2NN_trained/tyolov2NN_trained.trt'
DEF_CLASSES_FILE_PATH = './tyolov2NN_trained/voc.names'
CLASS_LABEL_TO_NOTIFY = "person" #LEDを点灯して通知したいクラスのラベル
PIN_NUM_LED_PLUS = 18 #LEDのアノード（+）に抵抗を経由してつながるピン番号（Rasbperry Pi GPIO番号であってボード上の番号ではない）

#YOLO v2のNNの本来の出力shape（ただしTRTの出力はこの通りではないことに注意）
output_shape_NN = (1, 125, 13, 13)

#訓練済みYOLO v2のNNの訓練時に決定されたアンカー
yolo_anchors = [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)]

#キャプチャ映像の上にメッセージを表示
def indicate_message_on_cap_img(cap_img_cv2, message, left, top):
    
    '''
    キャプチャ映像の上にメッセージを表示
    ＜入力＞
    ・cap_img_cv2：
    　土台となるキャプチャ画像
    ・message：
    　表示するメッセージ文字列
    ・left, top：
    　メッセージ文字列の左上の座標
    '''
    
    cv2.putText(cap_img_cv2, message, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
    # (0, 0, 255)なので文字色は赤  BGR

    
#検出結果をバウンディングボックスにして、キャプチャ映像の上に描画する
def draw_bboxes_on_cap_img(cap_img_cv2, bboxes, bboxes_score, bboxes_class_index, defined_classes):
    
    '''
    検出結果をバウンディングボックスにして、キャプチャ映像の上に描画する
    ＜入力＞
    ・cap_img_cv2：
    　OpenCVの画像オブジェクト。カメラのキャプチャ画像。BGR3チャンネル。(キャプチャ指定サイズh, 同じくw, 3)
    ・bboxes：
    　バウンディングボックス　(個数, 4)
    ・bboxes_score, bboxes_class_index：
    　各々のバウンディングボックスのスコアと推定クラスインデックス　(個数, )
    ・defined_classes：
    　定義済みクラスラベル一覧（List）
    '''
    
    cap_img_cv2_height, cap_img_cv2_width, _ = cap_img_cv2.shape
    
    for bbox, bbox_score, bbox_class_index in zip(bboxes, bboxes_score, bboxes_class_index):
        
        #bbox：このループでの描画対象bbox　(4,)
        
        bx, by, bw, bh = bbox
        
        #bboxの左右上下の線の座標を算出
        #ただし、土台であるキャプチャ映像をはみ出ないように               
        left = max(0, round(bx + 0.5))
        top = max(0, round(by + 0.5))
        right = min(cap_img_cv2_width, round(bx + bw + 0.5))
        bottom = min(cap_img_cv2_height, round(by + bh + 0.5))
        
        #実際にbboxを描画
        cv2.rectangle(cap_img_cv2, (left, top), (right, bottom), (0, 0, 255), 3)
        
        #推定クラスラベルとスコアの文字列表示
        class_label_str = defined_classes[bbox_class_index]
        score_str = str(round(bbox_score, 2))        
        class_label_score_str = class_label_str + " " + score_str
        indicate_message_on_cap_img(cap_img_cv2, class_label_score_str, right, top)
        #端末にも表示
        print("推定クラス：" + class_label_str + "　Score：" + str(score_str))


#OpenCVの画像オブジェクトを、YOLO v2用に変換
def convert_img_to_yolo2(img_cv2):
    
    '''
    OpenCVの画像オブジェクトを、YOLO v2用に変換
    ＜入力＞
    ・img_cv2：
    　OpenCVの画像オブジェクト。カメラのキャプチャ画像。BGR3チャンネル。ndarrayでshapeは(args.height, args.width, 3)。
    ＜出力＞
    ・img：
    　YOLO v2のNN入力用の画像オブジェクト。RGB3チャンネル。ndarrayでshapeは(1, 3, YOLO2_NN_INPUT_SIZE[0], 同じく[1])。
      メモリ配列をrow-majorで、画素値の型はfloat。
    '''
    
    #縦横サイズをYOLO2_NN_INPUT_SIZEに
    img = cv2.resize(img_cv2, YOLO2_NN_INPUT_SIZE) #(416, 416, 3)
    #CはBGRになっているので、RGBに
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    #画素値を符号なし整数値からfloatに
    img = img.astype(np.float32)
    #HWCをCHWに
    img = np.transpose(img, [2, 0, 1]) #(3, 416, 416)
    #NNはその入力データのaxis=0をバッチサイズと解釈するので、axis=0を追加
    img = np.expand_dims(img, axis=0) #(1, 3, 416, 416)
    #メモリ配列をrow-majorに
    img = np.array(img, dtype=np.float32, order='C')
    
    return img
    

#定義された20クラスラベルを取得（ローカルのDEF_CLASSES_FILE_PATHのファイルを読む）
def get_defined_classes():
    
    '''
    定義された20クラスラベルを取得（ローカルのDEF_CLASSES_FILE_PATHのファイルを読む）
    ＜出力＞
    ・classes
    　定義されたクラスラベルのList。要素は20個あるはず。
    '''
    #定義されたクラス名のListが返ってくる
    with open(DEF_CLASSES_FILE_PATH) as f:
        classes = [c.strip() for c in f.readlines()]
    
    #20クラスではない場合、AssertionErrorを出させる
    num_classes = len(classes)
    assert(num_classes==20)
    
    return classes

# Main function
def main():
    
    #コマンドライン引数の取得
    parser = argparse.ArgumentParser(description='Human Sensor with Tiny YOLO v2 Object Detector')
    
    parser.add_argument('--csi', \
        action='store_true', \
        help='Use CSI camera')
    parser.add_argument('--camera', '-c', \
        type=int, default=0, metavar='CAMERA_NUM', \
        help='Camera number')
    parser.add_argument('--width', \
        type=int, default=1280, metavar='WIDTH', \
        help='Capture width')
    parser.add_argument('--height', \
        type=int, default=720, metavar='HEIGHT', \
        help='Capture height')
    parser.add_argument('--flip_method', \
        type=int, default=0, metavar='FLIP_METHOD', \
        help='Capture rotation')
    parser.add_argument('--objth', \
        type=float, default=0.6, metavar='OBJ_THRESH', \
        help='Threshold of object confidence score (between 0 and 1)')
    parser.add_argument('--nmsth', \
        type=float, default=0.3, metavar='NMS_THRESH', \
        help='Threshold of NMS algorithm (between 0 and 1)')
    args = parser.parse_args()
    #csiのaction='store_true'については、以下を参照。「--csi」指定の場合True、その他はFalseとなる。
    #https://note.nkmk.me/python-argparse-bool/
    
    cap_img_width = args.width #指定されたキャプチャ映像のサイズ（幅）
    cap_img_height = args.height #指定されたキャプチャ映像のサイズ（幅）
    obj_threshold = args.objth
    nms_threshold = args.nmsth      
    
    print("args.csi:", args.csi)
    print("args.camera:", args.camera)
    print("args.width:", cap_img_width) 
    print("args.height:", cap_img_height)
    print("args.flip_method:", args.flip_method) 
    print("args.objth:", obj_threshold)
    print("args.nmsth:", nms_threshold)
    
    
    if args.csi==True:
        #CSIカメラを使用する場合
        
        #カメラ接続
        gst_cmd = GST_STR_CSI % (cap_img_width, cap_img_height, FPS, args.camera, args.flip_method,  cap_img_width, cap_img_height)
        print("gst_cmd:", gst_cmd)
        cap = cv2.VideoCapture(gst_cmd, cv2.CAP_GSTREAMER)
        
    else:
        #USBカメラを使用する場合
        
        #カメラ接続
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_img_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_img_height)
        
    
    try:
        
        if cap.isOpened()==False:
            print("カメラに接続できませんでした。終了します。")
            return    
        
        try:    
            #定義されたクラスラベルのリストを取得
            defined_classes = get_defined_classes()
            print("定義済みクラス取得\n", defined_classes)        
        except Exception as e:
            print("定義済みクラス情報を取得できませんでした。終了します。")
            return
        
        try:            
            #LEDを点灯させて通知したいクラスのインデックスを取得
            class_index_to_notify = defined_classes.index(CLASS_LABEL_TO_NOTIFY) 
        except Exception as e:
            class_index_to_notify = -1        
        
        print("LED点灯対象クラスラベル：", CLASS_LABEL_TO_NOTIFY)
        print("LED点灯対象クラスインデックス：", class_index_to_notify)
        
        if class_index_to_notify<0:
            print("警告：LED点灯対象クラスが定義済みクラスに存在しないため、LED点灯は行いません。")
        else:
            #LED制御に使用するピン番号に、Rasbperry Pi GPIO番号を使用
            GPIO.setmode(GPIO.BCM)
            #LED制御に使用するピンを出力ピンに　且つ初期状態を消灯
            GPIO.setup(PIN_NUM_LED_PLUS, GPIO.OUT, initial=GPIO.LOW)
            print("LEDアノード側の出力ピン(Rasbperry Pi GPIO番号:" + str(PIN_NUM_LED_PLUS) + ")　出力設定完了（初期状態:LOW）" )
            GPIO.output(PIN_NUM_LED_PLUS, GPIO.LOW) #一応念押しのLOW
                    
            
        #YOLO v2 NN出力データに基づきバウンディングボックスの生成と採用を行うクラスインスタンスを生成        
        yv2_bboxes_gen = Yolov2_bboxes_generator(
            yolo_anchors=yolo_anchors,
            obj_threshold=obj_threshold,
            nms_threshold=nms_threshold,
            input_size_NN=YOLO2_NN_INPUT_SIZE,
            defined_classes=defined_classes)
        print("Yolov2_bboxes_generatorインスタンス生成完了")


        #ONNXモデル格納ファイルからTensorRTエンジンを生成
        #TensorRTの実行コンテキスト生成
        #キャプチャ映像から物体検出し、その結果をウィンドウに表示
        
        #TensorRTエンジンを生成　TensorRTの実行コンテキスト生成
        with get_engine(ONNX_FILE_PATH, TRT_ENGINE_FILE_PATH) as engine, engine.create_execution_context() as context:

            #TensorRTにバッファを割り当てる            
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            print("TensorRTへのバッファ割り当て完了")

            frame_count = 0

            while True:

                #カメラで撮影
                #そのキャプチャ映像を1フレーム分読み込む。1フレームの経過時間はFPSの逆数。
                ret, cap_img_cv2 = cap.read() #(cap_img_height, cap_img_width, 3)            
                if ret!=True:
                    #ループの先頭へ
                    continue

                #1フレームのキャプチャ画像をYOLO v2のNNの入力形式に変換
                img_for_NN = convert_img_to_yolo2(cap_img_cv2) #(1, 3, 416, 416)
                
                #TensorRTで順伝播。YOLO v2のNNの順伝播出力を得る。
                #ただしデータ型は要素数1のList。
                #そのたった1つの要素はndarrayでshapeは(21125,)。
                inputs[0].host = img_for_NN
                output_from_trt = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

                #YOLO v2の本来の出力形状にする。
                output_NN =output_from_trt[0].reshape(output_shape_NN) #(1, 125, 13, 13)
                
                #NN出力を用いてバウンディングボックスの生成
                bboxes, bboxes_score, bboxes_class_index = yv2_bboxes_gen.generate_bboxes(output_NN, (cap_img_width, cap_img_height))
                
                #バウンディングボックスの描画とLED制御
                if bboxes is not None:
                    #バウンディングボックスがある
                    
                    if class_index_to_notify>=0:
                        #検出対象クラスが推定クラス中にあるかどうかでLED制御信号決定
                        if class_index_to_notify in bboxes_class_index:
                            #検出対象クラスが推定クラス中にある→LED点灯
                            GPIO.output(PIN_NUM_LED_PLUS, GPIO.HIGH)                            
                        else:
                            #検出対象クラスは推定クラス中に無い→LED消灯
                            GPIO.output(PIN_NUM_LED_PLUS, GPIO.LOW)
                                                
                    #バウンディングボックスの描画
                    draw_bboxes_on_cap_img(cap_img_cv2, bboxes, bboxes_score, bboxes_class_index, defined_classes)
                    
                else:
                    #バウンディングボックスは無い
                    
                    if class_index_to_notify>=0:
                        #LED制御信号　消灯
                        GPIO.output(PIN_NUM_LED_PLUS, GPIO.LOW)                        
                                        

                #キャプチャ映像とバウンディングボックスのウィンドウ表示
                cv2.imshow(WINDOW_NAME, cap_img_cv2)

                
                #ESCキーが押されたら終了
                keycode = cv2.waitKey(20)
                if keycode == 27: # ESC
                    break
                

                #ウィンドウの「閉じる」が押されたら終了
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
                    
                frame_count += 1
        
        
        if class_index_to_notify>=0:
            #LED制御信号　消灯
            GPIO.output(PIN_NUM_LED_PLUS, GPIO.LOW)
        
    
    except Exception as e:
        
        print("エラーまたは例外が発生しました。終了します。")
        print(e)
        traceback.print_exc()
        
    finally:
        
        #リソース解放
        #この順番が良いはず
        print("リソースを解放します。")
        GPIO.cleanup()
        cv2.destroyAllWindows()
        cap.release()
        print("リソースを解放しました。")        
    

if __name__ == "__main__":
    main()