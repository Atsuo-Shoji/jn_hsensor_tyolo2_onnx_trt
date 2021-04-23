#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#YOLO v2のNNの出力を受けて、バウンディングボックス（とそのスコア、推定クラス）を生成する
class Yolov2_bboxes_generator(object):
    

    def __init__(self,
                 yolo_anchors,
                 obj_threshold,
                 nms_threshold,
                 input_size_NN,
                 defined_classes
    ):
        
        '''
        ＜入力＞
        ・yolo_anchors：
        　YOLO V2のアンカー群。タプル(幅float,高さfloat)のリスト。通常は5個。
        ・object_threshold：
        　スコアでのバウンディングボックスの足切りの閾値。0～1のfloat。
        ・nms_threshold：
        　非最大値抑制の閾値。0～1のfloat。
        ・nn_input_size：
        　NNに入力された画像のサイズ。タプル(高さ、幅)。ちなみにNNに入力された画像のshapeは(N1, Ch3, H416, W416)。
        ・defined_classes：
        　定義されているクラスラベルのリスト。
        '''
        
        self._anchors = yolo_anchors
        self._object_threshold = obj_threshold
        self._nms_threshold = nms_threshold
        self._input_size_NN = input_size_NN
        self._defined_classes = defined_classes
        
        #1セルあたり生成するbboxesの個数（anchorsの個数と同じ）
        self._num_bboxes_cell = len(self._anchors)
        
        #定義されているクラス数
        self._num_classes = len(self._defined_classes)
        
        
    #YOLO v2のNNの出力を受けて、bboxes（とそのclass、score）を生成するpublic関数
    def generate_bboxes(self, output_from_NN, cap_img_size):
        
        '''
        YOLO v2のNNの出力を受けて、bboxes（とそのclass、score）を生成する。
        ＜入力＞
        ・output_from_NN：
        　NNの出力データ。通常は(1, 125, 13, 13)。
        ・cap_img_size：
        　キャプチャ画像のサイズ。タプル(幅, 高さ)。例(1280, 720)
        ＜出力＞
        ・bboxes：
        　bboxes（群）。ndarrayで(個数, 4)。
        ・bboxes_score：
        　上記bboxesの個々のbboxのscore。ndarrayで(個数,)。
        ・bboxes_class_index：
        　上記bboxesの個々のbboxの推定クラスインデックス。ndarrayで(個数,)。
        '''
        
        #bboxes生成のため、NNの出力データの形状を変更
        input_for_generation = self._reshape_output_for_generation(output_from_NN)
        
        #最終的なbboxes（とそのclass、score）の生成
        bboxes, bboxes_score, bboxes_class_index = self._generate_final_bboxes_attributes(input_for_generation, cap_img_size)
        
        return bboxes, bboxes_score, bboxes_class_index
        
    
    #bboxes生成のため、NNの出力データの形状を変更するprivate関数
    def _reshape_output_for_generation(self, output_from_NN):
        
        '''
        bboxes生成のため、NNの出力データの形状を変更する。
        ＜入力＞
        ・output_from_NN：
        　NNの出力データ。ndarrayで(1, アンカー個数x(5+定義済みクラス数), 13, 13)。通常は(1, 125, 13, 13)。
        ＜出力＞
        ・input_for_generation：
        　ndarrayで(13, 13, アンカー個数, 5+定義済みクラス数)。        　
        
        '''
        
        #output_from_NNを(13, 13, アンカー個数, 5+定義済みクラス数)のshapeに変換。通常は(13, 13, 5, 25)。        
        #2個の「13」は1画像中のセルの数、「アンカー個数」は1セルあたりで生成するbboxesの数で、anchorsの個数と等しい。
        #「5+定義済みクラス数」の「5」は、bbox1個の座標とサイズの4数値とscore。
        #「5+定義済みクラス数」の「定義済みクラス数」は、bbox1個の、定義済み各クラスについての確信度。通常は20クラス。
        
        output_ = np.transpose(output_from_NN, [0, 2, 3, 1])
        _, num_cells_height, num_cells_width, _ = output_.shape #ともに13のはず
        
        new_shape = (num_cells_height, num_cells_width, self._num_bboxes_cell, 4 + 1 + self._num_classes)
        
        input_for_generation = np.reshape(output_, new_shape)
        
        #print("input_for_generation:\n", input_for_generation)
                
        return input_for_generation
    
    
    #最終的なbboxes（とそのscore、推定クラス）を生成するprivate関数
    def _generate_final_bboxes_attributes(self, input_for_generate, cap_img_size):
        
        '''
        最終的なbboxes（とそのscore、推定クラス）を生成する。
        ＜入力＞
        ・input_for_generate：
        　入力データ。NNの出力のshapeを(13, 13, アンカー個数, 5+定義済みクラス数)に変換したndarray。
        ・cap_img_size：
        　キャプチャ画像のサイズ。タプル(幅, 高さ)。例(1280, 720)
        ＜出力＞
        ・bboxes：
        　bboxes（群）。ndarrayで(個数, 4)。scoreでの足切りやNMSを適用済み。
        ・bboxes_score：
        　上記bboxesの個々のbboxのscore。ndarrayで(個数,)。
        ・bboxes_class_index：
        　上記bboxesの個々のbboxの推定クラスインデックス。ndarrayで(個数,)。
        '''
        
        #規定量のbboxes、その各々のbboxの信頼度、定義済み各クラスの確信度を算出
        #「規定量」とは、1セルあたりのアンカー個数のこと。通常は、セル個数169 x アンカー個数5 = 845個。
        bboxes, bboxes_confidence, bboxes_class_probs = self._calc_bboxes_attributes(input_for_generate)
        
        #規定量のbboxes（とその属性）を、scoreを基準に足切り
        #選抜されたbboxes、その各々のscoreと推定クラスindexを返す
        bboxes, bboxes_score, bboxes_class_index = self._filter_bboxes(bboxes, bboxes_confidence, bboxes_class_probs)
        
        #bboxesの座標とサイズの単位を、cap_img_sizeに（ピクセルとなる）
        #次の非最大値抑制では、オリジナルの画像（キャプチャ画像）内のピクセル単位で処理を行うため
        cap_img_width, cap_img_height = cap_img_size
        cap_img_4_dims = [cap_img_width, cap_img_height, cap_img_width, cap_img_height] #bboxesの各列と対応させる
        bboxes = bboxes * cap_img_4_dims
        
        #非最大値抑制（「NMS」）
        #選抜されたbboxesを同一推定クラスごとにグループ分けし、そのグループごとにNMSを適用
        bboxes, bboxes_score, bboxes_class_index = self._nms_bboxes(bboxes, bboxes_score, bboxes_class_index)
        
        return bboxes, bboxes_score, bboxes_class_index
        
        
    #規定量のbboxes、その各々のbboxの信頼度とクラスの確信度を算出するprivate関数
    #「規定量」とは、「scoreで足切りしたりNMSで抑制したりする前の、仕様上フルの」と言う意味。
    #通常は、セル個数169 x 1セルあたりアンカー個数5 = 845個。
    def _calc_bboxes_attributes(self, input_for_calc):
        
        '''
        規定量のbboxes、その各々のbboxの信頼度とクラスの確信度を算出する。        
        「規定量」とは、「scoreで足切りしたりNMSで抑制したりする前の、仕様上フルの」と言う意味。ここでは1アンカーあたり1個のbboxを生成する。
        通常は、セル個数169 x 1セルあたりアンカー個数5 = 845個。
        ＜入力＞
        ・input_for_calc：
        　入力データ。NNの出力のshapeを(13, 13, アンカー個数, 5+定義済みクラス数)に変換したndarray。
        ＜出力＞
        ・bboxes：
        　bboxes（群）。ndarrayで(13, 13, アンカー個数, 4)。
        ・bboxes_confidence：
        　上記bboxesの個々のbboxの信頼度。ndarrayで(13, 13, アンカー個数, 1)。
        ・bboxes_class_probs：
        　上記bboxesの個々のbboxの各クラスの確信度。ndarrayで(13, 13, アンカー個数, 定義済みクラス数)。
          この「確信度」は確率ではない。全クラス分足して1にならない。これは定義済み全クラスがMECEとは限らないから。
          確率ではなく、単に0～1で正規化してあるだけ。
        '''
        
        #以降は、論文
        #https://arxiv.org/pdf/1612.08242.pdf
        #の、「Figure 3」に従った表記をする。
        
        ##全bboxes各々の5つの属性（座標とサイズと信頼度）、各クラスの確信度の算出##
        ##5つの属性とは、σ(tx),σ(tx),bw, bh,σ(to)のこと。
        
        #input_for_calcの、それら各々に割り当てられている列を分解して抽出
        
        #txとty
        txy = input_for_calc[...,:2] #txとty　(13, 13, アンカー個数, 2)
        #twとth
        twh = input_for_calc[...,2:4] #twとth　(13, 13, アンカー個数, 2)
        #to
        to = input_for_calc[...,4] #to　(13, 13, アンカー個数, )
        #最終軸の次元数が1なのでこの軸が消えてしまう。他と合わせるため、次元数1の最終軸を加える。
        to = np.expand_dims(to, axis=-1) #to　(13, 13, アンカー個数, 1)
        #各クラスの確信度の元になる数値
        prob_classes_raw = input_for_calc[...,5:] #各クラスの確信度の元になる数値　(13, 13, アンカー個数, 定義済みクラス数)
        
        #σ(tx),σ(tx),bw, bh,σ(to)、各クラスの確信度の算出
        #座標とサイズについては、単位は1セル（の幅と高さ）で、中心は所属セルの中心。
        
        #σ(tx)とσ(tx)
        bboxes_xy = self._sigmoid(txy) #σ(tx)とσ(tx)　(13, 13, アンカー個数, 2)
        #pwとpw　boxのサイズを計算するには、anchorsとtwhのかけ算をするが、かける相手twhと軸を合わせる（幅と高さでアダマール積を取る）。
        pwh = np.reshape(self._anchors, [1, 1, self._num_bboxes_cell, 2]) #pwとph　(1, 1, アンカー個数, 2)
        #bwとbh
        bwh = pwh * np.exp(twh) #bwとbh　(13, 13, アンカー個数, 2)
        #σ(to)
        bboxes_confidence = self._sigmoid(to) #σ(to)　(13, 13, アンカー個数, 1)
        #各クラスの確信度　確率ではない。softmaxを通さない。物体検出の定義済み全クラスは通常MECEではない。よって、0～1で正規化するだけ。
        bboxes_class_probs = self._sigmoid(prob_classes_raw) #(13, 13, アンカー個数, 定義済みクラス数)
        
        '''
        print("box_xy.shape:", box_xy.shape)
        print("box_xy:\n", box_xy)
        print()
        print("bwh.shape:", bwh.shape)
        print("bwh:\n", bwh)
        
        print("box_confidence.shape:", box_confidence.shape)
        print("box_confidence:\n", box_confidence)
        print()
        print("box_class_probs.shape:", box_class_probs.shape)
        print("box_class_probs:\n", box_class_probs)
        '''
        
        ##bboxの座標を変換 bx、byの算出##
        #①所属セル中の座標から、画像の左上を原点とした座標へ。cxとcyを各bboxの座標に加算する。
        #②bboxの座標を、その中心からbboxの左上の座標へ。
        
        #①所属セル中の座標から、NN入力画像の左上を原点とした座標へ。cxとcyを各bboxの座標に加算する。
        #例えば、セル(11,1)に注目する。左から12個目、上から2個目のセルである。
        #このセルには、アンカー個数と同じbboxesがある。各々のbboxの座標は（box_x, box_y）。これがアンカー個数個ある。
        #これらアンカー個数個のbboxesの座標を、等しく（box_x+11, box_y+1）する。このセルではcx=11、cy=1。
        #これを、全セル内の全bboxesに対して行う。        
        num_cells_height, num_cells_width, _, _ = input_for_calc.shape #ともに13のはず
        cx = np.tile(np.arange(0, num_cells_width), num_cells_width).reshape(-1, num_cells_width)
        cy = np.tile(np.arange(0, num_cells_height).reshape(-1, 1), num_cells_height)
        cx = cx.reshape(num_cells_height, num_cells_width, 1, 1).repeat(self._num_bboxes_cell, axis=-2)
        cy = cy.reshape(num_cells_height, num_cells_width, 1, 1).repeat(self._num_bboxes_cell, axis=-2)
        cxy = np.concatenate((cx, cy), axis=-1) #cxとcy　(13, 13, アンカー個数, 2)
        bboxes_xy += cxy
        
        #②bboxの座標を、その中心からbboxの左上の座標へ。
        bxy = bboxes_xy - (bwh / 2.0) #(13, 13, アンカー個数, 2)
        
        #これで、bx、by、bw、bhが出そろった。
        
        ##bboxの座標とサイズ（bx、by、bw、bh）の単位を、NN入力画像1枚にする
        #この時点では、単位は1セル（の幅、高さ）となっている。
        #これを、NN入力画像1枚（の幅、高さ）とする。通常、(416, 416)。
        bxy /= (num_cells_width, num_cells_height) #(13, 13, アンカー個数, 2)
        bwh /= (num_cells_width, num_cells_height) #(13, 13, アンカー個数, 2)
        
        #bboxの座標とサイズを1配列に統合
        bboxes = np.concatenate((bxy, bwh), axis=-1) #(13, 13, アンカー個数, 4)
        
        return bboxes, bboxes_confidence, bboxes_class_probs
    
    #規定量のbboxesを、scoreを基準に足切りするprivate関数
    def _filter_bboxes(self, bboxes, bboxes_confidence, bboxes_class_probs):
        
        '''
        規定量のbboxesを、scoreを基準に足切りする。
        ＜入力＞
        ・bboxes：
        　bboxes（群）。規定量ある。ndarrayで(13, 13, アンカー個数, 4)。
        ・bboxes_confidence：
        　上記bboxesの個々のbboxの信頼度。ndarrayで(13, 13, アンカー個数, 1)。
        ・bboxes_class_probs：
        　上記bboxesの個々のbboxの各クラスの確信度。ndarrayで(13, 13, アンカー個数, 定義済みクラス数)。
        ＜出力＞
        ・bboxes_qualified：
        　足切りで生き残って選抜されたbboxes。ndarrayで(選抜されたbboxes個数, 4)。
        ・bboxes_qualified_score：
        　上記bboxesの個々のbboxのscore。ndarrayで(選抜されたbboxes個数, )。
        ・bboxes_qualified_class_index：
        　上記bboxesの個々のbboxの推定クラスインデックス。ndarrayで(選抜されたbboxes個数, )。
          推定クラスインデックスは、「クラス毎のscore」 = 信頼度 x 各クラスの確信度　の最大値を与えるクラスのインデックス。
        '''
        
        ##各bboxの「score」を算出し、指定された閾値以上のbboxのみ採用        
        #各bboxの「score」は、「クラス毎のscore」 = 信頼度 x 各クラスの確信度　の最大値
        #各bboxのクラス毎のscore
        bboxes_scores_classes = bboxes_confidence * bboxes_class_probs #(13, 13, アンカー個数, 定義済みクラス数)
        #各bboxのscore
        bboxes_score = np.max(bboxes_scores_classes, axis=-1) #(13, 13, アンカー個数)
        #閾値self._object_thresholdと各bboxのscoreを比較し、bboxを選別
        #np.whereは、条件に合う配列のindexを返す。
        #その際、配列の各axisにおいて条件を満たすindexをndarrayにして、それらを全axisについて並べたtupleを返す。
        #例を下に書いた。
        bboxes_qualified_idx_tpl = np.where(bboxes_score >= self._object_threshold) #タプル。中の要素の配列のshapeは一概に決まらない。
        
        ##各bboxの推定クラスindexを決定
        #「クラス毎のscore」 = 信頼度 x 各クラスの確信度　の最大値を与えるクラスindex
        bboxes_class_idx = np.argmax(bboxes_scores_classes, axis=-1) #(13, 13, アンカー個数)
        
        '''
        print("bboxes_scores_classes:\n", bboxes_scores_classes)
        print()
        print("bboxes_score:\n", bboxes_score)
        print("bboxes_qualified_idx_tpl:\n", bboxes_qualified_idx_tpl)
        print()
        print("bboxes_class_idx:\n", bboxes_class_idx)        
        '''
        
        bboxes_qualified = bboxes[bboxes_qualified_idx_tpl]
        bboxes_qualified_score = bboxes_score[bboxes_qualified_idx_tpl]
        bboxes_qualified_class_index = bboxes_class_idx[bboxes_qualified_idx_tpl]
        
        
        '''
        例）
        足切りの対象となるbboxes_scoreのshapeは(13, 13, アンカー個数)、つまりaxisは3個。
        よって、タプルbboxes_qualified_idx_tplの要素数は3となる。
        
        ▼セル(11, 8)のbbox3番とセル(5, 4)のbbox1番の2個のbboxesが選抜された
        bboxes_qualified_idx_tpl[0] = [11, 5] ndarray
        bboxes_qualified_idx_tpl[1] = [8, 4] ndarray
        bboxes_qualified_idx_tpl[2] = [3, 1] ndarray
        bboxes[bboxes_qualified_idx_tpl].shape:(2, 4)
        bboxes_score[bboxes_qualified_idx_tpl].shape:(2,)
        bboxes_class_idx[bboxes_qualified_idx_tpl].shape:(2,)
        
        ▼セル(11, 8)のbbox3番のbboxが選抜された
        bboxes_qualified_idx_tpl[0] = [11] ndarray
        bboxes_qualified_idx_tpl[1] = [8] ndarray
        bboxes_qualified_idx_tpl[2] = [3] ndarray
        bboxes[bboxes_qualified_idx_tpl].shape:(1, 4)
        bboxes_score[bboxes_qualified_idx_tpl].shape:(1,)
        bboxes_class_idx[bboxes_qualified_idx_tpl].shape:(1,)
        
        ▼選抜されたbboxesが無かった
        bboxes_qualified_idx_tpl[0] = []
        bboxes_qualified_idx_tpl[1] = []
        bboxes_qualified_idx_tpl[2] = []
        bboxes[bboxes_qualified_idx_tpl].shape:(0, 4)
        bboxes_score[bboxes_qualified_idx_tpl].shape:(0,)
        bboxes_class_idx[bboxes_qualified_idx_tpl].shape:(0,)
        
        '''
        
        return bboxes_qualified, bboxes_qualified_score, bboxes_qualified_class_index
    
    
    #bboxesに対し非最大値抑制（「NMS」）を適用し、さらにbboxesを絞り込むprivate関数
    def _nms_bboxes(self, bboxes, bboxes_score, bboxes_class_index):
        
        '''
        bboxesに対し非最大値抑制（「NMS」）を適用し、さらにbboxesを絞り込む。
        ＜入力＞
        ・bboxes：
        　scoreでの足切り後、現時点で生き残っているbboxes。ndarrayで(残存bboxes個数, 4)。
        ・bboxes_score：
        　上記bboxesの個々のbboxのscore。ndarrayで(残存bboxes個数, )。
        ・bboxes_qualified_class_index：
        　上記bboxesの個々のbboxの推定クラスインデックス。ndarrayで(残存bboxes個数, )。
        ＜出力＞
        ・bboxes_through_nms_ary:
        　NMS適用後のbboxes。ndarrayで(適用後bboxes個数, 4)。
        ・bboxes_through_nms_score_ary：
        　上記bboxesの個々のbboxのscore。ndarrayで(適用後bboxes個数, )。
        ・bboxes_through_nms_class_index_ary：
        　上記bboxesの個々のbboxの推定クラスインデックス。ndarrayで(適用後bboxes個数, )。
        '''
        
        ##bboxesを同一推定クラスごとにグループ分けし、そのグループごとにNMSを適用
        
        #NMSを適用後のbboxesとそれらの属性を格納するリスト
        bboxes_through_nms,  bboxes_through_nms_score, bboxes_through_nms_class_index = [], [], []
        
        #各推定クラス毎にNMSを適用
        
        for a_class_index in set(bboxes_class_index):
            
            #a_class_indexは、このループのクラスのindex
            
            #a_class_indexのクラスを推定されたbboxesのindexを取得　NMS適用の対象
            idxes_bboxes_a_class = np.where(bboxes_class_index==a_class_index)
            
            #bboxes実体とそれらの属性（score、推定クラスindex）の抽出
            #a_class_indexのクラスを推定されたbboxes実体
            bboxes_a_class = bboxes[idxes_bboxes_a_class] #NMS適用対象　(対象bboxes個数, 4)
            #a_class_indexのクラスを推定されたbboxesの各bboxのscore
            bboxes_a_class_score = bboxes_score[idxes_bboxes_a_class] #NMS適用時の付属情報　(対象bboxes個数,)
            #a_class_indexのクラスを推定されたbboxesの各bboxの推定クラスindex
            bboxes_a_class_class_index = bboxes_class_index[idxes_bboxes_a_class] #(対象bboxes個数,)
            
            #NMS適用
            idxes_bboxes_a_class_through_nms = self._nms_bboxes_a_class(bboxes_a_class, bboxes_a_class_score)
            
            #NMSを適用後のbboxesとそれらの属性（score、推定クラスindex）を格納
            bboxes_through_nms.append(bboxes_a_class[idxes_bboxes_a_class_through_nms])
            bboxes_through_nms_score.append(bboxes_a_class_score[idxes_bboxes_a_class_through_nms])
            bboxes_through_nms_class_index.append(bboxes_a_class_class_index[idxes_bboxes_a_class_through_nms])
            
        if len(bboxes_through_nms)==0:
            #NMSを適用したら、生き残ったbboxesが無かった。
            #実際には、この関数に入って来た時に残存bboxes個数が0だった（NMS適用対象が無かった）場合のみ、ここに来る。
            #全推定クラスでNMSを適用したが選抜されたbboxesが無かった、はありえない。最低でもscore最大のbboxを1個選ぶ。
            #print("NMS Nothing")
            return None, None, None
        
        else:
            
            #各リストをndarrayにする。
            #各リストの中は、推定クラスごとに1個の1軸ndarrayが全推定クラス分並んでいる。
            #np.concatenateで、その1軸ndarrayの塊を壊して”平坦”な1軸ndarrayにする。
            
            #全推定クラス分NMSを適用した後のbboxes
            bboxes_through_nms_ary = np.concatenate(bboxes_through_nms) #(適用後bboxes個数, 4)
            #全推定クラス分NMSを適用した後のbboxesの各bboxのscore
            bboxes_through_nms_score_ary = np.concatenate(bboxes_through_nms_score) #(適用後bboxes個数,)
            #全推定クラス分NMSを適用した後のbboxesの各bboxの推定クラスindex
            bboxes_through_nms_class_index_ary = np.concatenate(bboxes_through_nms_class_index) #(適用後bboxes個数,)
            
            return bboxes_through_nms_ary, bboxes_through_nms_score_ary, bboxes_through_nms_class_index_ary
            
    
    #同一クラスと推定されたbboxesに対し非最大値抑制（「NMS」）を適用し、さらにbboxesを絞り込むprivate関数
    def _nms_bboxes_a_class(self, bboxes_a_class, bboxes_a_class_score):
        
        '''
        同一クラスと推定されたbboxesに対し非最大値抑制（「NMS」）を適用し、さらにbboxesを絞り込む。
        ＜入力＞
        ・bboxes_a_class：
        　scoreでの足切り後、現時点で生き残っているbboxesのうち、推定クラスが同一のもの。これらにNMSを適用してさらに絞り込む。
          ndarrayで(そのようなbboxes個数, 4)。
        ・bboxes_a_class_score：
        　上記bboxesの個々のbboxのscore。NMSで使用する尺度。ndarrayで(そのようなbboxes個数, )。
        ＜出力＞
        ・idxes_bboxes_adopted：
        　NMS適用後に生き残ったbboxesの（bboxes_a_classにおける）インデックス。ndarrayで(生き残ったbboxes個数,)。
        '''
        
        '''
        非最大値抑制（「NMS」）
        ”採用””不採用”フラグいずれでもないbboxesに対し、以下①→②を繰り返し行い、”採用””不採用”フラグを全bboxesに付ける
        ①bboxesのうち、scoreが最大値のbbox「基準bbox」に”採用”フラグ
        ②基準bboxとのIOUが閾値を超過しているbboxesに”不採用”フラグ
        →
        ”採用”フラグのbboxesのみを採用
        
        '''
        
        bboxes_left = bboxes_a_class[:, 0] #(n,)
        bboxes_top = bboxes_a_class[:, 1] #(n,)
        bboxes_w = bboxes_a_class[:, 2] #(n,)
        bboxes_h = bboxes_a_class[:, 3] #(n,)
        bboxes_right = bboxes_left + bboxes_w #(n,)
        bboxes_bottom = bboxes_top + bboxes_h #(n,)
        
        #各bboxの面積の配列
        bboxes_area = bboxes_w * bboxes_h  #(n,)
        
        #scoreの大きい順番に並べたbboxesのindex
        idxes_ordered_score = bboxes_a_class_score.argsort()[::-1] #(n,)
        
        #"採用"フラグが付いたbboxesのindexのリスト
        idxes_bboxes_adopted = []
        
        #以下のwhile各ループにて、対象となるbboxesの個数
        n_c = idxes_ordered_score.shape[0]
        
        while n_c > 0:
        
            #この1回のループで、対象bboxesは、
            #　scoreが最高のもの1個＝「基準bbox」：”採用”
            #　基準bboxとのIOU > self._nms_threshold　のbboxes：”不採用”
            #　上記以外（基準bboxとのIOU <= self._nms_threshold）のbboxes：フラグ無し　→　次のループはこれが対象bboxesに
            #の3つに分かれる。
            #以下、このループでの対象bboxesの個数をコメントにて「c」と表記する
            
            #print("beginning while n_c:", n_c)
            
            ##①bboxesのうち、scoreが最大値のbbox「基準bbox」に”採用”フラグ##
            
            #基準bboxのindex取得　単にidxes_ordered_scoreの先頭
            idx_std_bbox = idxes_ordered_score[0] #idx_std_bboxとは、基準bboxのindex
            
            #基準bboxに採用フラグを付加
            idxes_bboxes_adopted.append(idx_std_bbox)
            
            #基準bbox以外のbboxesのindex（scoreの大きい順に並んでいる）
            idxes_not_std_bboxes = idxes_ordered_score[1:] #(n_c-1,)
            
            ##②基準bboxとのIOUが閾値を超過しているbboxesに”不採用”フラグ##
            
            #基準bboxとその他の各bboxとのIOUを算出
            
            #基準bboxとその他の各bboxとの重なり領域の面積の算出
            
            #基準bboxとその他の各bboxの重なり領域の幅
            left_end_overlap = np.maximum(bboxes_left[idx_std_bbox], bboxes_left[idxes_not_std_bboxes]) #(n_c-1,)
            right_end_overlap = np.minimum(bboxes_right[idx_std_bbox], bboxes_right[idxes_not_std_bboxes]) #(n_c-1,)
            width_overlap = np.maximum(0.0, right_end_overlap - left_end_overlap + 1) #重なりが無い場合0にする (n_c-1,)
            
            #基準bboxとその他の各bboxの重なり領域の高さ
            top_end_overlap = np.maximum(bboxes_top[idx_std_bbox], bboxes_top[idxes_not_std_bboxes]) #(n_c-1,)
            bottom_end_overlap = np.minimum(bboxes_bottom[idx_std_bbox], bboxes_bottom[idxes_not_std_bboxes]) #(n_c-1,)
            height_overlap = np.maximum(0.0, bottom_end_overlap - top_end_overlap + 1) #重なりが無い場合0にする (n_c-1,)
            #あるbboxのy座標においてはtop<bottomであることに注意。bottomの方がtopよりy座標の数値は大きい。
            
            #基準bboxとその他の各bboxとの重なり領域「AND」の面積
            overlap = width_overlap * height_overlap #(n_c-1,)
            
            #基準bboxとその他の各bboxとの和集合「OR」の面積
            union = bboxes_area[idx_std_bbox] + bboxes_area[idxes_not_std_bboxes] - overlap #(n_c-1,)
            
            #IOU
            iou = overlap / union #(n_c-1,)
            
            #基準bboxとのIOUが閾値超過のbboxesは”不採用”、閾値以下のbboxesはフラグ無し
            #しかし実際には”不採用”フラグを付加ということはせず、フラグ無しとなるべきbboxesを抽出し、次のループに送る。
            #np.whereの戻り値は、条件に合う各axisのindexのndarrayを全axis分並べてタプルにしたもの。
            #iouのshapeは(n_c,)の1軸ベクトルなので、タプルの要素は1個のndarrayだけ。下記[0]はそれを抽出。
            idxes_no_flag = np.where(iou <= self._nms_threshold)[0] #(フラグ無しbboxes個数,)
            idxes_ordered_score = idxes_ordered_score[idxes_no_flag + 1] #この「+ 1」の意味は下記にまとめて記す。
            n_c = idxes_ordered_score.shape[0]
            
            
            '''
            「idxes_no_flag + 1」の「 + 1」について。
            iou（及びその算出元となったoverlap, unionなど）の中の要素の並び順は、
            idxes_ordered_scoreの中の要素の並び順と同じで、scoreの大きい順。
            ただし、iouなどは、基準bboxが要素中に無い。shapeは(n_c-1,)。他のbboxesの基準bboxとの”関わり”を保持する配列だから。
            一方、idxes_ordered_scoreは、基準bboxが要素中にあり、当然その位置は先頭（index=0）である。shapeは(n_c,)。
            つまり、両者は、中身は並び順が同じだが、”インデックスが1個だけずれている”配列同志である。
            idxes_ordered_score[0]に相当するものはiouにはなく、
            idxes_ordered_score[1]に相当するものはiou[0]
            idxes_ordered_score[2]に相当するものはiou[1]
            　・・・
            そして、idxes_no_flagはiou配列の中身のインデックスであることから、+1している。
            
            '''
        
        #リストであるidxes_bboxes_adoptedをndarray1軸ベクトルに
        idxes_bboxes_adopted = np.array(idxes_bboxes_adopted)
        
        return idxes_bboxes_adopted
                
    
    def _sigmoid(self, v):
        
        #v：スカラーかndarray。
        
        ret = 1.0 / (1.0 + np.exp(-v))
        
        return ret
    
    
    @property
    def anchors(self):
        return self._anchors
    
    @property
    def object_threshold(self):
        return self._object_threshold
    
    @property
    def nms_threshold(self):
        return self._nms_threshold
    
    @property
    def input_size_NN(self):
        return self._input_size_NN
    
    @property
    def defined_classes(self):
        return self._defined_classes
    
    @property
    def num_defined_classes(self):
        return self._num_classes
    
    @property
    def num_bboxes_cell(self):
        return self._num_bboxes_cell 