# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1163, 820)
        MainWindow.setMaximumSize(QtCore.QSize(1600, 850))
        MainWindow.setStyleSheet("#MainWindow{border-image:url(./img_resource/bg.jpg);}")

        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(940, 730, 211, 31))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(190, 10, 521, 71))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 0, 511, 61))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.TimeSecondLCD = QtWidgets.QLCDNumber(self.groupBox_3)
        self.TimeSecondLCD.setGeometry(QtCore.QRect(440, 20, 64, 36))
        self.TimeSecondLCD.setMaximumSize(QtCore.QSize(100, 16777215))
        self.TimeSecondLCD.setObjectName("TimeSecondLCD")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(200, 20, 81, 36))
        self.label_10.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_10.setObjectName("label_10")
        self.FmRateLCD = QtWidgets.QLCDNumber(self.groupBox_3)
        self.FmRateLCD.setGeometry(QtCore.QRect(90, 20, 71, 36))
        self.FmRateLCD.setMaximumSize(QtCore.QSize(100, 16777215))
        self.FmRateLCD.setObjectName("FmRateLCD")
        self.TimeMinuteLCD = QtWidgets.QLCDNumber(self.groupBox_3)
        self.TimeMinuteLCD.setGeometry(QtCore.QRect(360, 20, 64, 36))
        self.TimeMinuteLCD.setMaximumSize(QtCore.QSize(100, 16777215))
        self.TimeMinuteLCD.setObjectName("TimeMinuteLCD")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(160, 20, 31, 36))
        self.label_13.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_13.setObjectName("label_13")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(0, 20, 90, 36))
        self.label_12.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_12.setObjectName("label_12")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(350, 30, 16, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(24)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.TimeHourLCD = QtWidgets.QLCDNumber(self.groupBox_3)
        self.TimeHourLCD.setGeometry(QtCore.QRect(280, 20, 71, 36))
        self.TimeHourLCD.setMaximumSize(QtCore.QSize(100, 16777215))
        self.TimeHourLCD.setObjectName("TimeHourLCD")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(430, 30, 16, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(24)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.Msg = QtWidgets.QTextEdit(self.centralwidget)
        self.Msg.setGeometry(QtCore.QRect(260, 730, 661, 31))
        self.Msg.setObjectName("Msg")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(250, 190, 681, 531))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.groupBox_5 = QtWidgets.QGroupBox(self.frame_5)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 10, 661, 511))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.Camera_2 = QtWidgets.QLabel(self.groupBox_5)
        self.Camera_2.setGeometry(QtCore.QRect(10, 20, 640, 480))
        self.Camera_2.setText("")
        self.Camera_2.setObjectName("Camera_2")
        self.frame_9 = QtWidgets.QFrame(self.centralwidget)
        self.frame_9.setGeometry(QtCore.QRect(20, 190, 221, 51))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.Blink_freq = QtWidgets.QLCDNumber(self.frame_9)
        self.Blink_freq.setGeometry(QtCore.QRect(80, 10, 71, 31))
        self.Blink_freq.setMinimumSize(QtCore.QSize(0, 30))
        self.Blink_freq.setMaximumSize(QtCore.QSize(90, 16777215))
        self.Blink_freq.setObjectName("Blink_freq")
        self.label_22 = QtWidgets.QLabel(self.frame_9)
        self.label_22.setGeometry(QtCore.QRect(160, 10, 61, 30))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.frame_9)
        self.label_23.setGeometry(QtCore.QRect(0, 0, 81, 51))
        self.label_23.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_23.setObjectName("label_23")
        self.frame_16 = QtWidgets.QFrame(self.centralwidget)
        self.frame_16.setGeometry(QtCore.QRect(20, 240, 221, 51))
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.label_27 = QtWidgets.QLabel(self.frame_16)
        self.label_27.setGeometry(QtCore.QRect(0, 0, 81, 51))
        self.label_27.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_27.setObjectName("label_27")
        self.label_19 = QtWidgets.QLabel(self.frame_16)
        self.label_19.setGeometry(QtCore.QRect(160, 10, 61, 30))
        self.label_19.setObjectName("label_19")
        self.Yawn_freq = QtWidgets.QLCDNumber(self.frame_16)
        self.Yawn_freq.setGeometry(QtCore.QRect(80, 10, 70, 30))
        self.Yawn_freq.setMinimumSize(QtCore.QSize(0, 30))
        self.Yawn_freq.setMaximumSize(QtCore.QSize(90, 16777215))
        self.Yawn_freq.setObjectName("Yawn_freq")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(710, 10, 451, 71))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.groupBox_4 = QtWidgets.QGroupBox(self.frame_6)
        self.groupBox_4.setGeometry(QtCore.QRect(0, 0, 461, 71))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.btntestcamera = QtWidgets.QPushButton(self.groupBox_4)
        self.btntestcamera.setGeometry(QtCore.QRect(15, 20, 95, 41))
        self.btntestcamera.setObjectName("btntestcamera")
        self.btn_testvideo = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_testvideo.setGeometry(QtCore.QRect(240, 20, 95, 41))
        self.btn_testvideo.setObjectName("btn_testvideo")
        self.btn_start = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_start.setGeometry(QtCore.QRect(130, 20, 94, 41))
        self.btn_start.setObjectName("btn_start")
        self.btnexit = QtWidgets.QPushButton(self.groupBox_4)
        self.btnexit.setGeometry(QtCore.QRect(350, 20, 94, 41))
        self.btnexit.setObjectName("btnexit")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(930, 130, 231, 591))
        font = QtGui.QFont()
        font.setFamily("黑体")
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.State_record = QtWidgets.QTextEdit(self.groupBox)
        self.State_record.setGeometry(QtCore.QRect(10, 20, 211, 561))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.State_record.setFont(font)
        self.State_record.setObjectName("State_record")
        self.frame_17 = QtWidgets.QFrame(self.centralwidget)
        self.frame_17.setGeometry(QtCore.QRect(20, 340, 221, 51))
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.shake_LCD = QtWidgets.QLCDNumber(self.frame_17)
        self.shake_LCD.setEnabled(True)
        self.shake_LCD.setGeometry(QtCore.QRect(80, 10, 71, 31))
        self.shake_LCD.setMaximumSize(QtCore.QSize(1677, 16777215))
        self.shake_LCD.setObjectName("shake_LCD")
        self.label_32 = QtWidgets.QLabel(self.frame_17)
        self.label_32.setGeometry(QtCore.QRect(160, 10, 61, 30))
        self.label_32.setObjectName("label_32")
        self.label_34 = QtWidgets.QLabel(self.frame_17)
        self.label_34.setGeometry(QtCore.QRect(0, 0, 81, 51))
        self.label_34.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_34.setObjectName("label_34")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(20, 390, 221, 51))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.Danger_LCD = QtWidgets.QLCDNumber(self.frame)
        self.Danger_LCD.setEnabled(True)
        self.Danger_LCD.setGeometry(QtCore.QRect(80, 10, 71, 31))
        self.Danger_LCD.setMaximumSize(QtCore.QSize(1677215, 16777215))
        self.Danger_LCD.setObjectName("Danger_LCD")
        self.label_33 = QtWidgets.QLabel(self.frame)
        self.label_33.setGeometry(QtCore.QRect(160, 10, 61, 30))
        self.label_33.setObjectName("label_33")
        self.label_35 = QtWidgets.QLabel(self.frame)
        self.label_35.setGeometry(QtCore.QRect(0, 0, 81, 51))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_35.setFont(font)
        self.label_35.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_35.setObjectName("label_35")
        self.frame_8 = QtWidgets.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(20, 290, 221, 51))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.label_28 = QtWidgets.QLabel(self.frame_8)
        self.label_28.setGeometry(QtCore.QRect(0, 0, 81, 51))
        self.label_28.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_28.setObjectName("label_28")
        self.Nod_LCD = QtWidgets.QLCDNumber(self.frame_8)
        self.Nod_LCD.setGeometry(QtCore.QRect(80, 10, 71, 31))
        self.Nod_LCD.setObjectName("Nod_LCD")
        self.label_29 = QtWidgets.QLabel(self.frame_8)
        self.label_29.setGeometry(QtCore.QRect(160, 10, 61, 30))
        self.label_29.setObjectName("label_29")
        self.label_31 = QtWidgets.QLabel(self.frame_8)
        self.label_31.setGeometry(QtCore.QRect(190, 50, 61, 30))
        self.label_31.setObjectName("label_31")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(10, 100, 181, 61))
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(16)
        self.label_11.setFont(font)
        self.label_11.setTextFormat(QtCore.Qt.AutoText)
        self.label_11.setObjectName("label_11")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(690, 80, 471, 51))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.VideoPath = QtWidgets.QTextEdit(self.frame_2)
        self.VideoPath.setGeometry(QtCore.QRect(60, 10, 191, 31))
        self.VideoPath.setObjectName("VideoPath")
        self.BtnReadvideo = QtWidgets.QPushButton(self.frame_2)
        self.BtnReadvideo.setGeometry(QtCore.QRect(260, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(9)
        self.BtnReadvideo.setFont(font)
        self.BtnReadvideo.setObjectName("BtnReadvideo")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(20, 5, 41, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 41, 21))
        self.label_2.setObjectName("label_2")
        self.BtnRecord = QtWidgets.QPushButton(self.frame_2)
        self.BtnRecord.setGeometry(QtCore.QRect(370, 10, 94, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        self.BtnRecord.setFont(font)
        self.BtnRecord.setObjectName("BtnRecord")
        self.frame_18 = QtWidgets.QFrame(self.centralwidget)
        self.frame_18.setGeometry(QtCore.QRect(700, 130, 231, 51))
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.label_30 = QtWidgets.QLabel(self.frame_18)
        self.label_30.setGeometry(QtCore.QRect(0, 0, 81, 51))
        self.label_30.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_30.setObjectName("label_30")
        self.Emotion_pred = QtWidgets.QLCDNumber(self.frame_18)
        self.Emotion_pred.setGeometry(QtCore.QRect(74, 10, 81, 31))
        self.Emotion_pred.setMaximumSize(QtCore.QSize(100, 50))
        self.Emotion_pred.setObjectName("Emotion_pred")
        self.Emotion = QtWidgets.QLabel(self.frame_18)
        self.Emotion.setGeometry(QtCore.QRect(160, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Emotion.setFont(font)
        self.Emotion.setText("")
        self.Emotion.setObjectName("Emotion")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(10, 0, 131, 101))
        self.logo.setText("")
        self.logo.setObjectName("logo")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 440, 221, 321))
        font = QtGui.QFont()
        font.setFamily("黑体")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_pred_img = QtWidgets.QLabel(self.groupBox_2)
        self.label_pred_img.setGeometry(QtCore.QRect(1, 20, 211, 300))
        self.label_pred_img.setText("")
        self.label_pred_img.setObjectName("label_pred_img")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(190, 80, 501, 101))
        self.widget.setObjectName("widget")
        self.frame_4 = QtWidgets.QFrame(self.widget)
        self.frame_4.setGeometry(QtCore.QRect(340, 0, 151, 51))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_21 = QtWidgets.QLabel(self.frame_4)
        self.label_21.setGeometry(QtCore.QRect(0, 0, 81, 51))
        self.label_21.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_21.setObjectName("label_21")
        self.State = QtWidgets.QLabel(self.frame_4)
        self.State.setGeometry(QtCore.QRect(80, 0, 61, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.State.setFont(font)
        self.State.setObjectName("State")
        self.frame_11 = QtWidgets.QFrame(self.widget)
        self.frame_11.setGeometry(QtCore.QRect(20, 0, 151, 101))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.frame_12 = QtWidgets.QFrame(self.frame_11)
        self.frame_12.setGeometry(QtCore.QRect(0, 0, 151, 51))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.label_24 = QtWidgets.QLabel(self.frame_12)
        self.label_24.setGeometry(QtCore.QRect(0, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.Eyes_state = QtWidgets.QLabel(self.frame_12)
        self.Eyes_state.setGeometry(QtCore.QRect(80, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Eyes_state.setFont(font)
        self.Eyes_state.setObjectName("Eyes_state")
        self.frame_10 = QtWidgets.QFrame(self.frame_11)
        self.frame_10.setGeometry(QtCore.QRect(0, 50, 151, 51))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.label_17 = QtWidgets.QLabel(self.frame_10)
        self.label_17.setGeometry(QtCore.QRect(0, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.Mouth_state = QtWidgets.QLabel(self.frame_10)
        self.Mouth_state.setGeometry(QtCore.QRect(80, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Mouth_state.setFont(font)
        self.Mouth_state.setObjectName("Mouth_state")
        self.frame_7 = QtWidgets.QFrame(self.widget)
        self.frame_7.setGeometry(QtCore.QRect(340, 50, 151, 51))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.PERCLOS = QtWidgets.QLCDNumber(self.frame_7)
        self.PERCLOS.setGeometry(QtCore.QRect(80, 10, 65, 31))
        self.PERCLOS.setMaximumSize(QtCore.QSize(100, 50))
        self.PERCLOS.setObjectName("PERCLOS")
        self.label_16 = QtWidgets.QLabel(self.frame_7)
        self.label_16.setGeometry(QtCore.QRect(0, 9, 81, 41))
        self.label_16.setMinimumSize(QtCore.QSize(0, 40))
        self.label_16.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_16.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_16.setObjectName("label_16")
        self.frame_13 = QtWidgets.QFrame(self.widget)
        self.frame_13.setGeometry(QtCore.QRect(170, 0, 161, 101))
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.frame_14 = QtWidgets.QFrame(self.frame_13)
        self.frame_14.setGeometry(QtCore.QRect(0, 0, 161, 51))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.label_25 = QtWidgets.QLabel(self.frame_14)
        self.label_25.setGeometry(QtCore.QRect(0, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.Head_state = QtWidgets.QLabel(self.frame_14)
        self.Head_state.setGeometry(QtCore.QRect(80, 0, 81, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Head_state.setFont(font)
        self.Head_state.setObjectName("Head_state")
        self.frame_15 = QtWidgets.QFrame(self.frame_13)
        self.frame_15.setGeometry(QtCore.QRect(0, 50, 161, 51))
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.label_26 = QtWidgets.QLabel(self.frame_15)
        self.label_26.setGeometry(QtCore.QRect(0, 0, 71, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.Danger_state = QtWidgets.QLabel(self.frame_15)
        self.Danger_state.setGeometry(QtCore.QRect(80, 0, 81, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Danger_state.setFont(font)
        self.Danger_state.setObjectName("Danger_state")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1163, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        ##按钮样式设置
        self.btn_start.setStyleSheet(''' 
                             QPushButton
                             {text-align : center;
                             background-color : white;
                             font: bold;
                             border-color: gray;
                             border-width: 2px;
                             border-radius: 10px;
                             padding: 6px;
                             height : 14px;
                             border-style: outset;
                             font : 14px;}
                             QPushButton:pressed
                             {text-align : center;
                             background-color : light gray;
                             font: bold;
                             border-color: gray;
                             border-width: 2px;
                             border-radius: 10px;
                             padding: 6px;
                             height : 14px;
                             border-style: outset;
                             font : 14px;}
                             ''')

        self.btnexit.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : red;
                     font: bold;
                     color:white;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
        self.btntestcamera.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : yellow;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
        self.btn_testvideo.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     color:black;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
        self.BtnRecord.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     color:black;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
        self.BtnReadvideo.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     color:black;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "优行·危险驾驶行为分析系统V1.0"))
        self.label_9.setText(_translate("MainWindow", "不该小组·优行驾驶"))
        self.groupBox_3.setTitle(_translate("MainWindow", "系统信息"))
        self.label_10.setText(_translate("MainWindow", "当前时间："))
        self.label_13.setText(_translate("MainWindow", "FPS"))
        self.label_12.setText(_translate("MainWindow", "当前帧频："))
        self.label_14.setText(_translate("MainWindow", ":"))
        self.label_15.setText(_translate("MainWindow", ":"))
        self.groupBox_5.setTitle(_translate("MainWindow", "实时预测"))
        self.label_22.setText(_translate("MainWindow", "次数/分"))
        self.label_23.setText(_translate("MainWindow", "眨眼频率"))
        self.label_27.setText(_translate("MainWindow", "哈欠频率"))
        self.label_19.setText(_translate("MainWindow", "次数/分"))
        self.groupBox_4.setTitle(_translate("MainWindow", "菜单"))
        self.btntestcamera.setText(_translate("MainWindow", "测试相机"))
        self.btn_testvideo.setText(_translate("MainWindow", "测试视频"))
        self.btn_start.setText(_translate("MainWindow", "开始运行"))
        self.btnexit.setText(_translate("MainWindow", "退出程序"))
        self.groupBox.setTitle(_translate("MainWindow", "状态记录"))
        self.label_32.setText(_translate("MainWindow", "次数/分"))
        self.label_34.setText(_translate("MainWindow", "摇头频率"))
        self.label_33.setText(_translate("MainWindow", "帧率/分"))
        self.label_35.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">危险行为频率</span></p></body></html>"))
        self.label_28.setText(_translate("MainWindow", "点头频率"))
        self.label_29.setText(_translate("MainWindow", "次数/分"))
        self.label_31.setText(_translate("MainWindow", "次数/分"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:20pt; font-weight:600; color:#0000ff;\">优行·驾驶</span></p></body></html>"))
        self.BtnReadvideo.setText(_translate("MainWindow", "读取文件"))
        self.label.setText(_translate("MainWindow", "文件"))
        self.label_2.setText(_translate("MainWindow", "路径"))
        self.BtnRecord.setText(_translate("MainWindow", "保存记录"))
        self.label_30.setText(_translate("MainWindow", "预测情绪"))
        self.groupBox_2.setTitle(_translate("MainWindow", "情绪预测"))
        self.label_21.setText(_translate("MainWindow", "当前状态"))
        self.State.setText(_translate("MainWindow", "清醒"))
        self.label_24.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">眼部状态</span></p></body></html>"))
        self.Eyes_state.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">嘴部状态</span></p></body></html>"))
        self.Mouth_state.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_16.setText(_translate("MainWindow", "PERCLOS"))
        self.label_25.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">头部状态</span></p></body></html>"))
        self.Head_state.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_26.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">危险行为</span></p></body></html>"))
        self.Danger_state.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
