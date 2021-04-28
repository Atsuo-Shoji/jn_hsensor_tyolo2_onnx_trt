# Jetson Nano ～ Tiny YOLO v2で対人センサー


## 概要

Jetson Nano上で訓練済みTiny YOLO v2を使用し、人間を検出したら赤色LEDを点灯させる”簡易対人センサー”を作りました。<BR>
※訓練済みTiny YOLO v2は、[ONNX公式リポジトリ](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2/model)で公開されているものです。自作ではありません。

<BR>

| 「person」を検出<br>赤色LEDが点灯 | 「person」を検出せず<br>赤色LEDは点灯しない |
|      :---:     |      :---:      |
|![person_on_8sec_crop](https://user-images.githubusercontent.com/52105933/115808263-25366880-a425-11eb-9716-ab5faf962d15.gif)|![bus_off_8sec_crop](https://user-images.githubusercontent.com/52105933/115808494-89592c80-a425-11eb-9c88-988b99ae256a.gif)|

※理論の説明は基本的にしていません。他のリソースを参考にしてください。<br>
&nbsp;&nbsp; ネットや書籍でなかなか明示されておらず、私自身が実装に際し情報収集や理解に不便を感じたものを中心に記載しています。

<br>

### 動作

事前に、Jetson Nanoにカメラ（CSIでもUSBでも可）と赤色LEDが適切に接続されているとします。<br>

![全体接続図_80](https://user-images.githubusercontent.com/52105933/115812412-4cdcff00-a42c-11eb-9c30-04f74e7655ff.png)

アプリケーションを走らせます。カメラが撮影を開始します。<BR>
カメラで撮影した映像に対し訓練済みモデルが推論した結果である、バウンディングボックスとその推定クラス及びスコアを、その映像上に表示します。<BR>
![推論結果表示_80](https://user-images.githubusercontent.com/52105933/115976577-4c6a7280-a5aa-11eb-95f6-a84fa7a759e8.png) <BR>
同時に、推定クラス「person」を検出している間は赤色LEDを点灯します。<br>
「person」を検出していない間は赤色LEDを消灯します。<br>
※本稿では、「CSIカメラ」とは「Raspberry Pi camera v2」のことです。


| 事象 | 動作 |
|:---     | :---        |
|「person」を検出|![LEDイラスト_on_30](https://user-images.githubusercontent.com/52105933/115810047-2e750480-a428-11eb-91ba-87389dbec789.png) <BR>赤色LED点灯|
|「person」を検出せず<BR>・「person」以外のみ検出<BR>・何も検出せず|![LEDイラスト_off_30](https://user-images.githubusercontent.com/52105933/115810074-3e8ce400-a428-11eb-974a-8828113adfed.png) <BR>赤色LED消灯|

#### 定義済みの20クラス

訓練済みのTiny YOLO v2モデルを使用しています。<BR>
Pascal VOC Challengeで使用された、以下の20クラスで訓練されています。<BR>
従って、検出したバウンディングボックスの推定クラスも、以下のいずれかになります。<BR>
LEDを点灯させる「person」もこの20クラスのうちの1つです。
```
 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 
 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
 'sofa', 'train', 'tvmonitor' 
```

<br>

### 推論の大まかな流れと、その中での自作部分

以下のように推論し、推論結果（バウンディングボックスとその推定クラス、スコア）を生成/表示します。<br>
この流れのうち、カメラからのキャプチャ映像を訓練済みTiny YOLO v2のONNXモデルに入力し、TensorRTを通して出力する「順伝播出力」の部分は、他者製作のもの（※1）を使用させていただきました。<br>

![推論の流れ_80](https://user-images.githubusercontent.com/52105933/115960307-9cfeb300-a54b-11eb-9c98-cb973369840d.png)

※1：詳細は、後の「ディレクトリ・ファイル構成」参照のこと
<BR>

### ディレクトリ・ファイル構成
<BR>

| ファイル名 | 製作者または取得場所 |概要|
|:---  | :---  | :---  |
|human_sensor_tyolo2_onnx_trt.py|自作|アプリケーションのmain。<br>カメラでの撮影、キャプチャ映像と推論結果の表示、LED制御を行っている。|
|Yolov2_bboxes_generator.py|自作|YOLO v2のNNの順伝播出力を受けて、バウンディングボックスの生成などの推論を行う。<BR>バウンディングボックスの座標やサイズの算出、スコアでの足切り、非最大値抑制など。|
|./tyolov2NN_trained<BR>/tyolov2NN_trained.onnx|ONNXの公式レポジトリ中の[ここ](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2/model)の「tinyyolov2-8.tar.gz」|Pascal VOC Datasetで訓練済みのTiny YOLO v2のNNのONNXモデル。|
|./tyolov2NN_trained<BR>/voc.names|[これ](https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names)|定義済み20クラスラベル。<BR>Pascal VOC Challengeで使用されたもの。|
|./tyolov2NN_trained<BR>/tyolov2NN_trained.trt|TensorRTが自動生成|上記訓練済みTiny YOLO v2のNNのONNXモデルのTensorRTプラン。<BR>環境で初回実行時に自動生成される。|
|get_engine.py|NVIDIA|ONNXモデルからTensorRTプランの自動生成。<br>TensorRTランタイムの実行の仲介。|
|common.py|NVIDIA|TensorRTランタイムの推論実行。|

<BR>

これより以下は、<br>
カメラでキャプチャした映像を訓練済みモデルを通してバウンディングボックスとその推定クラスを表示する「推論」<br>
LEDをJetson NanoにつないでPythonから制御可能にする「LED設置」<br>
に分けて記述します。
<br><br>

## 推論 ～ バウンディングボックスや推定クラスなどの表示まで
<br>

ここらへんを担当するのは「Yolov2_bboxes_generator.py」です。<BR>
各ndarrayのshapeに至るまでコメントを多めに細かく書いたので、基本的にはそれらコメントを参照してください。<BR>
ここでは、コメントではわかりづらそうなところだけ記述します。<BR>

### キャプチャ映像取得から推論表示までの流れ

![bboxes決定と表示とLED制御_流れ_75](https://user-images.githubusercontent.com/52105933/116016431-8eadb580-a677-11eb-8ccd-86eee6d80f2d.png)
<br>

### バウンディングボックスの座標とサイズ算出の詳細

YOLO v2 のNNの推論時の順伝播出力は、定義済みクラス数を20とすると、shape(1, 125, 13, 13)のndarrayです。<br>
（ただしTensorRTプランからの出力の場合、shape(21125,)のndarrayが1個だけあるListです。）<br>
これをtransposeとreshapeして、<br>
shape(13, 13, 5, (5 + 20))にします。意味は以下の通りです。<br>
（以降、[論文「YOLO9000_Better, Faster, Stronger」](https://arxiv.org/pdf/1612.08242.pdf)の「Figure 3」に沿った表記をします）<br>

![出力データ構造_80](https://user-images.githubusercontent.com/52105933/115961134-0b457480-a550-11eb-89d2-0e222f7fd7d7.png)
<br>

このshape(13, 13, 5, 25)のndarrayに対して、以下の演算を加えます。<br>
演算結果がバウンディングボックスの座標とサイズです。<br>

![bbox座標とサイズ算出演算_80](https://user-images.githubusercontent.com/52105933/115941026-9dad3000-a4de-11eb-9609-72417587c503.png)
<br>

この時点では、バウンディングボックスの座標とサイズの単位は「セル」です。<br>
この後に各バウンディングボックスのスコアでの足切りをし、その後に非最大値抑制を適用します。<br>
この非最大値抑制の前に、単位を「ピクセル（画素）」に変換します。
<br><br>

## LED設置 ～ Pythonから制御可能な状態にするまで
<br>

Jetson Nano本体につなげるところから、Pythonから制御可能な状態にするまで、記述します。<BR>

### Jetson Nanoとの接続
<br>

電子工作用ブレッドボードに赤色LEDと抵抗を挿します。<BR>
オス－メスのジャンパー線2本を、それぞれ、ブレッドボードとJetson NanoのJ41のしかるべき場所に挿します。<br>

#### 使用したもの（実際に購入した型番などは、別途記述します）

- 電子工作用ブレッドボード<br>
- 赤色LED<br>
標準電流：20mA<br>
VF：2.1V<br>
- オス－メスのジャンパー線2本<br>
- 電気抵抗<br>
抵抗値：75Ω<br>
定格電力：0.25Ｗ<br>

#### 回路・配線

以下のように設置・配線しました。

- Jetson NanoのJ41の「12」がプラスで、「GND」がマイナス<br>
- ジャンパー線のメス側をJetson NanoのJ41の「12」に、オス側を抵抗－LEDアノードに<br>
- もう1本のジャンパー線のメス側をJetson NanoのJ41の「GND」に、オス側をLEDカソードに<br>

![回路・配線図_80](https://user-images.githubusercontent.com/52105933/115948515-80418b80-a509-11eb-8d70-b214f890f164.png)

#### 必要な抵抗・実際の電流の計算

Jetson NanoのGPIOの電圧は3.3V<br>
LEDの標準電流は20mAでVFは2.1V<br>
なので、必要な抵抗は、以下のように計算できます。<BR>
```
必要な抵抗
(Jetson NanoのGPIOの電圧3.3V - LEDのVF2.1V) ÷ 20mA = 60Ω
```

しかし、60Ωの抵抗は販売されていなかったため、75Ωの抵抗を使用することにしました。<BR>

すると、実際にLEDに流れる電流は、以下のように計算できます。<BR>
```
実際にLEDに流れる電流
(Jetson NanoのGPIOの電圧3.3V - LEDのVF2.1V) ÷ 75Ω = 16mA
```
<BR>

### Pythonから制御可能な状態にする
<BR>

LEDを物理的に正しく取り付けた後は、これをPythonから点灯/消灯できる状態にまで持っていきます。<br>

#### Jetson GPIO Library　のインストール

PythonでLEDを制御するには、Jetson GPIO Libraryが必要なので、これをインストールします。<br>

※既にJetson.GPIOがインストールされている場合<BR>
私の環境では、既にJetson.GPIO 2.0.16がインストールされていました。<br>
おそらくそのような人は他にも大勢いる、と思われます（特に最近のJetPackの場合）。<br>
ですが、私は敢えてインストールしました。<br>
後述するカスタムルール「99-gpio.rules」ファイルが環境内のどこにも無く、そのままではPythonから動作させることができなかったからです。<br>
インストールせず、Jetson GPIO Libraryのgitのリポジトリからそのファイルを取得して所定の場所に置くだけで動作するかもしれませんが、試していないのでわかりません。<br>

Jetson GPIO Libraryをgitからcloneしてセットアッププログラムを実行します。<br>
端末で以下をします。<br>
```
git clone https://github.com/NVIDIA/jetson-gpio.git
cd jetson-gpio
sudo python3 setup.py install
```

#### アプリケーションからGPIOポートを操作する権限をユーザーに付与

Jetson GPIO Libraryを使用するグループ「gpio」を作り、そこに（普段使っている）ユーザーを追加します。<br>
端末で以下をします。<br>
```
sudo groupadd -f -r gpio
sudo usermod -a -G gpio (普段使っているユーザー名）
```

このグループに権限を付与します。<br>
（カレントディレクトリが「Jetson GPIO Library　のインストール」の後のまま、即ち、cloneしてきた「jetson-gpio」ディレクトリ直下、という前提です）<br>
カスタムルール「99-gpio.rules」ファイルを所定の場所にコピーし、ルールを反映します。<br>
端末で以下をします。<br>
```
sudo cp lib/python/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

終わったら再起動が必要です。<br>
再起動後、PythonからLEDの制御が可能になっています。<br>
LEDを1秒間隔で点滅させるサンプルプログラムを実行して、確認してもよいでしょう。<BR>
端末で以下をします。<br>
```
python3 ./jetson-gpio/samples/simple_out.py
```

<br>

## 実行確認環境と実行の方法

本リポジトリは、他者に提供するためまたは実行してもらうために作ったものではないため、記載しません。<br>

ただし、LED周りで実際に購入した電子工作部品については、同様のことを考えている方の一助になればと思い、以下に記載します。<BR>

#### ＜参考＞実際に購入した電子工作部品

| 商品名 | 写真 |購入場所|補足|
|:---  | :---:  | :---  | :---  |
|ミニブレッドボード<BR>BB601（スケルトン）|![P-05156_25](https://user-images.githubusercontent.com/52105933/115956916-7f751d80-a53a-11eb-8c80-aeed7c6a4f27.jpg)|[秋月電子通商の商品ページ](https://akizukidenshi.com/catalog/g/gP-05156/)|－|
|ブレッドボード・<br>ジャンパーワイヤ（オス－メス）<BR>　15cm（黒）（10本入）|![C-08932_25](https://user-images.githubusercontent.com/52105933/115957031-23f75f80-a53b-11eb-9caf-c53e610fa108.jpg)|[秋月電子通商の商品ページ](https://akizukidenshi.com/catalog/g/gC-08932/)|－|
|3mm赤色LED　660nm<BR>OSR7CA3131A（10個入）|![I-04780_25](https://user-images.githubusercontent.com/52105933/115957259-2e662900-a53c-11eb-9f3f-50f8c90ed616.jpg)|[秋月電子通商の商品ページ](https://akizukidenshi.com/catalog/g/gI-04780/)|標準電流：20mA<BR>VF：2.1V|
|カーボン抵抗（炭素皮膜抵抗）<BR>1/4W 75Ω（100本入）|![R-25750_25](https://user-images.githubusercontent.com/52105933/115957363-ea275880-a53c-11eb-8b2e-80a54cb4122c.jpg)|[秋月電子通商の商品ページ](https://akizukidenshi.com/catalog/g/gR-25750/)|－|

<br>

## その他

本リポジトリは、NVIDIA作成のオープンソースのプログラムを利用しており、以下の条件でライセンスされています。<BR>
```
Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.

NOTICE TO LICENSEE:

This source code and/or documentation ("Licensed Deliverables") are
subject to NVIDIA intellectual property rights under U.S. and
international Copyright laws.

These Licensed Deliverables contained herein is PROPRIETARY and
CONFIDENTIAL to NVIDIA and is being provided under the terms and
conditions of a form of NVIDIA software license agreement by and
between NVIDIA and Licensee ("License Agreement") or electronically
accepted by Licensee.  Notwithstanding any terms or conditions to
the contrary in the License Agreement, reproduction or disclosure
of the Licensed Deliverables to any third party without the express
written consent of NVIDIA is prohibited.

NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
OF THESE LICENSED DELIVERABLES.

U.S. Government End Users.  These Licensed Deliverables are a
"commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
1995), consisting of "commercial computer software" and "commercial
computer software documentation" as such terms are used in 48
C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
U.S. Government End Users acquire the Licensed Deliverables with
only those rights set forth herein.

Any use of the Licensed Deliverables in individual and commercial
software must include, in the user documentation and internal
comments to the code, the above Disclaimer and U.S. Government End
Users Notice.
```

<br><br>

※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。