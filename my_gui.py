import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtGui

from qtui import Ui_Dialog


class MyMainForm(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.btn_open.clicked.connect(self.open_image)
        self.btn_submit.clicked.connect(self.close)

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.original_image.width(), self.original_image.height())
        self.original_image.setPixmap(jpg)

    def display(self):
        # 利用line Edit控件对象text()函数获取界面输入
        username = self.usertext.text()
        password = self.pwdtext.text()
        # 利用text Browser控件对象setText()函数设置界面显示
        self.display.setText("登录成功!\n" + "用户名是: " + username + ",密码是： " + password)


if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
