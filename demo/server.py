from Pinyin2Hanzi import DefaultDagParams, dag, is_pinyin
from flask import request, send_from_directory
from flask import Flask
import os

from src.utils.BrailleProcessor import BrailleProcessor

app = Flask(__name__)

dp = BrailleProcessor('206.h5', 'index_value.txt')

@app.route("/upload", methods=['GET','POST'])
def get_frame():
    upload_file = request.files['file']
    file_name = upload_file.filename
    file_path = r'./'
    if upload_file:
        file_paths = os.path.join(file_path, 'temp.png')
        upload_file.save(file_paths)
        res = get_text_from_pic(file_paths)
        return res
    return ''


def get_text_from_pic(img_path):
    img, segments = dp.get_segments(img_path, 0)
    if len(segments) < 1:
        return

    s = ''
    i = 0
    index = 0
    sentence = []
    word = ''
    special_letter_dic = {'g': 'j', 'k': 'q', 'h': 'x', 'i': 'yi', 'uo': 'wo', 'iao': 'yao', 'iu': 'you', 'u': 'wu'}

    # 处理各个切片
    while i < len(segments):
        # area = [0, 103, 1, 65]
        # [y起始,x起始]
        area = segments[i]
        img_segment = dp.get_specified_area_in_img(img, area)

        # 预测
        index += 1
        label_index = dp.predict(img_segment)
        label = dp.get_label_by_index(label_index)
        # print('label:', label_index)
        print('识别切片', i, ': ', label)

        # 切片为声调，开始拼接
        if label in ['一声', '二声', '三声', '四声']:
            # 特殊转换的拼音
            if word in ['i', 'uo', 'iao', 'iu', 'u']:
                word = special_letter_dic.get(word)
            if len(word) == 1:
                # 需补充i
                if word in ['zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'n']:
                    word = word + 'i'

            # 拼音规范性检查
            if is_pinyin(word):
                sentence.append(word)
                print('拼音为:', word)
                print('句子为:', sentence)
            else:
                print('该拼音不符合规范:' + word)
            word = ''

        # 切片为空白区域，跳过
        elif label_index == 54:
            word = ''

        #  切片为拼音
        else:
            # g/k/h 在韵母 i/u/ü 时，变读为j/q/x
            if label[0] in ['i', 'u', 'v']:
                if len(word) == 1 and word[0] in ['g', 'k', 'h']:
                    word = special_letter_dic.get(word)

            word = word + label
        i += 1

    # 拼音转文字
    dag_params = DefaultDagParams()
    res = dag(dag_params, sentence, path_num=2)

    res = res[1].path
    for item in res:
        s += item
    print('识别结果:', s)
    return s


if __name__ == '__main__':
    app.run(host='0.0.0.0')
