import datetime

from PyQt5 import QtWidgets

from MainWindow import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer, QCoreApplication, QDateTime
from PyQt5.QtGui import QPixmap, QImage
import qimage2ndarray
from torch.autograd import *
from detection import *
from ssd_net_vgg import *
from voc0712 import *
import torch
import torch.nn as nn
import numpy as np
import cv2
import utils
import torch.backends.cudnn as cudnn
import time
import StartDect
class CameraShow(QMainWindow,Ui_MainWindow):

    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return
    def __init__(self,parent=None):
        super(CameraShow,self).__init__(parent)
        self.setupUi(self)
        self.PrepCamera()

        self.CallBackFunctions()
        self.showTime()
        self.Timer=QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)

        self.video_flg = True
    #     prepare
    def PrepCamera(self):
        try:
            self.camera=cv2.VideoCapture(0)
            self.Image_num = 0
            self.Msg.clear()
            self.Msg.append('Oboard camera connected.')
            self.Msg.setPlainText()
            self.showTime()
        except Exception as e:
            self.Msg.clear()
            self.Msg.append(str(e))

    def CallBackFunctions(self):
        self.btntestcamera.clicked.connect(self.testCamera)
        # self.StopBt.clicked.connect(self.StopCamera)
        self.btn_start.clicked.connect(self.StartDection)
        self.btnexit.clicked.connect(self.ExitApp)

    #显示时间
    def showTime(self):
        # time = QDateTime.currentDateTime()
        now_time = datetime.datetime.now()

        self.timer = QTimer()
        self.timer.timeout.connect(self.showTime)
        self.timer.start(1000)
        # timeDisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")
        hour=now_time.strftime('%H')
        minute=now_time.strftime('%M')
        second=now_time.strftime('%S')
        self.TimeHourLCD.display(hour)
        self.TimeMinuteLCD.display(minute)
        self.TimeSecondLCD.display(second)


    def ColorAdjust(self, img):
        try:
            B = img[:, :, 0]
            G = img[:, :, 1]
            R = img[:, :, 2]

            # B.astype(cv2.PARAM_UNSIGNED_INT)
            # G.astype(cv2.PARAM_UNSIGNED_INT)
            # R.astype(cv2.PARAM_UNSIGNED_INT)

            img1 = img
            img1[:, :, 0] = B
            img1[:, :, 1] = G
            img1[:, :, 2] = R
            return img1
        except Exception as e:
            self.Msg.setPlainText(str(e))

#打开相机
    def testCamera(self):
        if self.Timer.isActive() == False:
            flag = self.camera.open(0)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.Timer.start(1)
                self.timelb = time.clock()
                self.btntestcamera.setText(u'关闭相机')
                self.btn_start.setEnabled(False)
                self.btn_testvideo.setEnabled(False)
        else:
            # self.Timer.stop()
            self.camera.release()
            self.Camera.clear()
            self.btntestcamera.setText(u'打开相机')
            self.btn_start.setEnabled(True)
            self.btn_testvideo.setEnabled(True)


    def TimerOutFun(self):
        success,img=self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            #self.Image=img
            self.DispImg()
            self.showTime()
            self.Image_num += 1
            if self.Image_num%10==9:
                frame_rate=10/(time.clock()-self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb=time.clock()
                #size=img.shape
                # self.ImgWidthLCD.display(self.camera.get(3))
                # self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.Msg.clear()
            self.Msg.setPlainText('Image obtaining failed.')




    def StartDection(self):
        if self.Timer.isActive() == False:
            flag = self.camera.open(0)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.Timer.start(1)
                self.timelb = time.clock()
                self.btn_start.setText(u'停止运行')
                self.btntestcamera.setEnabled(False)
                self.btn_testvideo.setEnabled(False)
                success,img=self.camera.read()
                # if success:
                #     img2=StartDect.StartDect(img)
                #     self.show_img(img2)
                #     # self.Camera.setPixmap(QPixmap(img2))
                #     # self.Camera.show()
                #     QApplication.processEvents()
                # else:
                #     self.Msg.clear()
                #     self.Msg.setPlainText('Image obtaining failed.')

                while self.video_flg:
                    # 按帧读取图像
                    ret, self.img_scr = self.camera.read()
                    # opencv中图像为BGR，这里转为RGB
                    # 因为我的SSD训练时用的是RGB图像，顺序错误会影响检测准确性
                    self.img_scr = StartDect.StartDect(self.img_scr)
                    # 更新显示图像
                    self.show_img(self.img_scr)

                    # 强制更新UI
                    # 如果没有，界面就‘假死’了，因为一直处于循环里
                    QApplication.processEvents()




        else:
            # self.Timer.stop()
            self.camera.release()
            self.Camera.clear()
            self.btn_start.setText(u'开始运行')
            self.btn_testvideo.setEnabled(True)
            self.btntestcamera.setEnabled(True)

    def show_img(self, img):
        showImg = QImage(img.data, img.shape[1], img.shape[0],
                         img.shape[1] * 3,  # 每行数据个数，3通道 所以width*3
                         QImage.Format_RGB888)
        self.Camera.setPixmap(QPixmap.fromImage(showImg))  # 展示图片

    #display
    def DispImg(self):

        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.Camera.setPixmap(QPixmap(qimg))
        self.Camera.show()
    # def StopCamera(self):
    #     if self.StopBt.text()=='暂停':
    #         self.StopBt.setText('继续')
    #         self.RecordBt.setText('保存')
    #         self.Timer.stop()
    #     elif self.StopBt.text()=='继续':
    #         self.StopBt.setText('暂停')
    #         self.RecordBt.setText('录像')
    #         self.Timer.start(1)

    #退出程序
    def ExitApp(self):
        self.Timer.Stop()
        self.camera.release()
        self.Msg.setPlainText('Exiting the application..')
        QCoreApplication.quit()



    #     关闭 X
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.camera.isOpened():
                self.camera.release()
            if self.Timer.isActive():
                self.Timer.stop()
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui=CameraShow()
    ui.show()
    sys.exit(app.exec_())