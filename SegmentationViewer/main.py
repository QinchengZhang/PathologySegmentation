# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-14 10:32:38
LastEditors: TJUZQC
LastEditTime: 2020-09-15 16:01:31
Description: None
'''
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

import Viewer

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Viewer.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
