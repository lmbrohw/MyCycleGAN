from tkinter import *

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import ImageTk, Image

def openImage():
    global img_png
    img_path = filedialog.askopenfilenames(initialdir='./datasets/test/A')
    if len(img_path):
        img_path = img_path[0]
        img_open = Image.open(img_path)
    w, h = img_open.size
    img_open = resize(w, h, w_box, h_box, img_open)
    img_png = ImageTk.PhotoImage(img_open)
    label_img_left['image'] = img_png
    global input_path
    global onshow_path
    input_path = img_path
    onshow_path = img_path
    input_content.set('已选择图像 {}'.format(img_path))

def resize(w, h, w_box, h_box, pil_image):
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
    '''
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def transfer(model_name):
    # thresh = s1.get()
    print('---------已选择模型{}-----------'.format(model_name))
    global onshow_path
    if onshow_path == 'NOINPUT':
        output_content.set('没有选择输入图片')
    model_name = model_name
    input_name = onshow_path.split('/')[-1]
    result_path = 'demo_files/results/{}_{}_fake.png'.format(model_name, input_name.split('.')[0])
    print('图像已保存至', result_path)
    # if not os.path.exists(result_path):  # 没有生成过
    #     # 运行模型
    #     demoTest(model_name, onshow_path, input_name, thresh)
    #     print(onshow_path)
    # 运行模型
    # demoTest(model_name, onshow_path, input_name, thresh)
    print(onshow_path)

    # 展示图片
    global result_png
    result_png = showImage(result_path)
    label_img_right['image'] = result_png
    output_content.set('生成模型 {} 的结果'.format(model_name))

def showImage(img_path):
    img_open = Image.open(img_path)
    w, h = img_open.size
    img_open = resize(w, h, w_box, h_box, img_open)
    img_png = ImageTk.PhotoImage(img_open)
    return img_png

if __name__ == '__main__':
    window = Tk()
    window.geometry('1280x600')
    window.title('demo')
    window["background"] = "#C9C9C9"
    # 预置
    flag_sign = False
    input_path = 'NOINPUT'
    onshow_path = 'NOINPUT'
    text_ori = '寒蝉凄切,对长亭晚,骤雨初歇'
    # 期望图像显示的大小
    w_box = 400
    h_box = 400
    # 标签
    lbl = Label(window, text="人像动漫化模型demo", font=("Arial Bold", 18))
    lbl.place(y=10, x=450, width=300, height=40)
    # 选择图片
    btn = Button(window, text="选择图片", command=openImage)
    btn.place(y=500, x=300, width=200, height=40)

    # 设置按钮
    btn_set1 = Button(window, text="原模型", command=lambda:transfer(model_name='dataset1'))
    btn_set1.place(y=100, x=40, width=130, height=40)
    btn_set2 = Button(window, text="更换数据集", command=lambda:transfer(model_name='dataset2'))
    btn_set2.place(y=200, x=40, width=130, height=40)
    # 交互显示区域
    input_content = tk.StringVar(value='没有输入图像')  # 这个就是我们创建的容器，类型为字符串类型
    input = tk.Label(window, compound=CENTER, textvariable=input_content, wraplength = 400)  # 用textvariable与容器绑定
    input.place(y=450, x=200, width=400)

    output_content = tk.StringVar(value='没有生成结果')
    output = tk.Label(window, compound=CENTER, textvariable=output_content, wraplength = 400)
    output.place(y=450, x=700, width=400)
    # 图像展示
    label_img_left = Label(window, image='')
    label_img_left.place(y=100, x=200)
    label_img_right = Label(window, image='')
    label_img_right.place(y=100, x=700)

    # main
    window.mainloop()