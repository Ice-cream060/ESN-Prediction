import sys
import os
import re
import PyQt5
import pandas as pd
import numpy as np
import reservoirpy as rpy
plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QGroupBox, QMessageBox, QFrame,
                             QTabWidget, QPlainTextEdit, QSplitter, QGraphicsDropShadowEffect,
                             QDoubleSpinBox, QSizeGrip)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QMouseEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from main import DataPipeline, ESNPredictor



class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(45)
        self.setStyleSheet("background-color: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(8)

        btn_size = 13
        style_template = """
            QPushButton {{
                background-color: {color}; 
                border-radius: 6px; 
                border: 1px solid {border};
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

        self.btn_close = QPushButton()
        self.btn_close.setFixedSize(btn_size, btn_size)
        self.btn_close.setStyleSheet(style_template.format(color="#FF5F56", border="#E0443E", hover="#FF3B30"))
        self.btn_close.clicked.connect(self.parent.close)

        self.btn_min = QPushButton()
        self.btn_min.setFixedSize(btn_size, btn_size)
        self.btn_min.setStyleSheet(style_template.format(color="#FFBD2E", border="#DEA127", hover="#FF9500"))
        self.btn_min.clicked.connect(self.parent.showMinimized)

        self.btn_max = QPushButton()
        self.btn_max.setFixedSize(btn_size, btn_size)
        self.btn_max.setStyleSheet(style_template.format(color="#27C93F", border="#1AAB29", hover="#34C759"))
        self.btn_max.clicked.connect(self.toggle_max_restore)

        layout.addWidget(self.btn_close)
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)

        layout.addStretch(1)
        self.title_label = QLabel("ESN 神经动力学预测系统")
        self.title_label.setStyleSheet(
            "color: #666666; font-family: 'Segoe UI'; font-weight: bold; font-size: 12px; letter-spacing: 2px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        right_spacer = QWidget()
        right_spacer.setFixedWidth(btn_size * 3 + 16)
        layout.addWidget(right_spacer)

        self.start_pos = None

    def toggle_max_restore(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos() - self.parent.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton and self.start_pos:
            if self.parent.isMaximized():
                self.parent.showNormal()
            self.parent.move(event.globalPos() - self.start_pos)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.toggle_max_restore()



class AcademicDarkCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=120):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#000000')
        fig.patch.set_alpha(0.0)

        self.ax1 = fig.add_subplot(111)
        self.ax1.set_facecolor('#000000')
        self.ax2 = self.ax1.twinx()

        for ax in [self.ax1, self.ax2]:
            ax.tick_params(colors='#888888', direction='in', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')

        self.ax1.spines['left'].set_color('#333333')
        self.ax1.spines['right'].set_visible(False)
        self.ax2.spines['right'].set_color('#333333')
        self.ax2.spines['left'].set_visible(False)

        self.ax1.grid(True, linestyle='-', alpha=0.1, color='#FFFFFF')
        self.label_font = {'family': 'serif', 'color': '#FFFFFF', 'size': 10, 'weight': 'bold'}

        super(AcademicDarkCanvas, self).__init__(fig)
        self.setStyleSheet("background-color: transparent;")


def create_glass_card():
    card = QFrame()
    card.setStyleSheet("""
        QFrame {
            background-color: rgba(25, 25, 25, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }
    """)
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(20)
    shadow.setColor(QColor(0, 0, 0, 150))
    shadow.setOffset(0, 4)
    card.setGraphicsEffect(shadow)
    return card



class ESNDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 去除边框与实现圆角背景透明
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1366, 850)
        self.df_user = None

        self.base_qss = """
                    QMainWindow { background-color: transparent; }
                    QWidget { color: #E0E0E0; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; }
                    QWidget#RootWidget { 
                        background-color: #0A0A0A; border: 1px solid #2A2A2A; border-radius: 12px; 
                    } 

                    QGroupBox { 
                        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 4px; 
                        margin-top: 20px; padding: 15px 10px 10px 10px; color: #888888;
                    }
                    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #FFFFFF; font-weight: bold;}

                    QTabWidget::pane { border: 1px solid #333333; border-radius: 4px; background: rgba(0,0,0,0); }
                    QTabBar::tab { background: #0A0A0A; color: #666666; border: 1px solid #333333; 
                                   padding: 6px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                    QTabBar::tab:selected { background: #1A1A1A; color: #FFFFFF; border-bottom-color: #1A1A1A; font-weight: bold;}

                    QPlainTextEdit { background-color: rgba(0,0,0,0.3); border: 1px solid #333333; border-radius: 4px; padding: 5px; color: #FFFFFF; }
                    QPlainTextEdit:focus { border: 1px solid #888888; }

                    QPushButton#ActionBtn {
                        background-color: #FFFFFF; color: #000000; 
                        border-radius: 4px; padding: 12px; font-weight: bold; font-size: 14px; letter-spacing: 1px;
                    }
                    QPushButton#ActionBtn:hover { background-color: #CCCCCC; }
                    QPushButton#ActionBtn:disabled { background-color: #333333; color: #777777; }

                    QPushButton#ToggleBtn {
                        background-color: transparent; color: #888888; border: 1px solid #333333; border-radius: 4px; padding: 5px 10px;
                    }
                    QPushButton#ToggleBtn:hover { color: #FFFFFF; border: 1px solid #FFFFFF; }

                    QPushButton#SubBtn {
                        background-color: #1A1A1A; border: 1px solid #333333; border-radius: 4px; padding: 8px; color: #E0E0E0; font-weight: bold;
                    }
                    QPushButton#SubBtn:hover { background-color: #333333; border: 1px solid #888888; }

                    QSlider::groove:horizontal { border-radius: 2px; height: 3px; background: #333333; }
                    QSlider::handle:horizontal { 
                        background: #FFFFFF; width: 14px; height: 14px; 
                        margin: -5px 0; border-radius: 7px; border: 1px solid #000000;
                    }
                    QSlider::handle:horizontal:hover { background: #AAAAAA; }

                    /* 基础样式：去掉了固定的 font-size，将由缩放引擎接管 */
                    QLabel#MetricVal { font-weight: 900; background: transparent; }
                    QLabel#ErrorVal { font-weight: bold; color: #FFFFFF; background: transparent; }
                    QLabel#CardTitle { color: #777777; background: transparent; }

                    QMessageBox { background-color: #111111; }
                    QMessageBox QPushButton { background-color: #FFFFFF; color: #000000; padding: 5px 15px; border-radius: 3px; font-weight: bold; }
                """
        self.setStyleSheet(self.base_qss)

        self.initUI()

    def initUI(self):
        root_widget = QWidget()
        root_widget.setObjectName("RootWidget")
        self.setCentralWidget(root_widget)

        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        root_layout.addWidget(self.title_bar)

        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        root_layout.addWidget(content_widget)

        top_bar = QHBoxLayout()
        self.btn_toggle = QPushButton("◀ 收起控制面板")
        self.btn_toggle.setObjectName("ToggleBtn")
        self.btn_toggle.setFixedWidth(120)
        self.btn_toggle.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.btn_toggle)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        self.splitter = QSplitter(Qt.Horizontal)

        # ================= 左侧控制面板 =================
        self.left_panel = create_glass_card()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)

        data_group = QGroupBox("1. 数据载入与解析")
        data_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        tab_manual = QWidget()
        manual_layout = QVBoxLayout(tab_manual)
        manual_layout.addWidget(QLabel("静息心率 (BPM) 序列："))
        self.text_hr = QPlainTextEdit()
        self.text_hr.setPlaceholderText("至少输入10个数据...")
        manual_layout.addWidget(self.text_hr)

        manual_layout.addWidget(QLabel("睡眠质量 (Score) 序列："))
        self.text_ss = QPlainTextEdit()
        self.text_ss.setPlaceholderText("至少输入10个数据...")
        manual_layout.addWidget(self.text_ss)
        manual_layout.addStretch()

        tab_csv = QWidget()
        csv_layout = QVBoxLayout(tab_csv)
        self.btn_upload = QPushButton("选择本地 CSV 文件")
        self.btn_upload.setObjectName("SubBtn")
        self.btn_upload.clicked.connect(self.load_data)
        self.lbl_file_status = QLabel("等待导入...")
        self.lbl_file_status.setStyleSheet("color: #777; font-size: 12px;")
        csv_layout.addWidget(self.btn_upload)
        csv_layout.addWidget(self.lbl_file_status)
        csv_layout.addStretch()

        self.tabs.addTab(tab_manual, "快捷输入")
        self.tabs.addTab(tab_csv, "CSV 导入")
        data_layout.addWidget(self.tabs)

        # --- 新增功能：保存数据的按钮 ---
        self.btn_save = QPushButton("导出 / 保存当前数据")
        self.btn_save.setObjectName("SubBtn")
        self.btn_save.clicked.connect(self.save_current_data)
        data_layout.addWidget(self.btn_save)

        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        param_group = QGroupBox("2. ESN 动力学调参")
        param_layout = QVBoxLayout()
        guide_lbl = QLabel("<div style='color:#777; font-size:10px;'>参数直接映射储备池非线性动力学属性。</div>")
        param_layout.addWidget(guide_lbl)

        self.slider_units, self.val_units = self.create_slider(param_layout, "储备池节点", 10, 500, 100, 1)
        self.slider_sr, self.val_sr = self.create_slider(param_layout, "谱半径", 50, 150, 99, 100)
        self.slider_lr, self.val_lr = self.create_slider(param_layout, "泄露率", 1, 100, 10, 100)
        self.slider_sparsity, self.val_sparsity = self.create_slider(param_layout, "稀疏度", 1, 50, 10, 100)
        self.slider_ridge, self.val_ridge = self.create_slider(param_layout, "岭回归", 1, 100, 10, 10)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        left_layout.addSpacing(10)
        self.btn_run = QPushButton("启动推演")
        self.btn_run.setObjectName("ActionBtn")
        self.btn_run.clicked.connect(self.run_prediction)
        left_layout.addWidget(self.btn_run)
        left_layout.addStretch()

        # ================= 右侧数据面板 =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        cards_layout = QHBoxLayout()

        pred_card = create_glass_card()
        pred_layout = QHBoxLayout(pred_card)
        hr_box = QVBoxLayout()
        hr_box.addWidget(QLabel("t+1 静息心率", objectName="CardTitle"))
        self.lbl_pred_hr = QLabel("--", objectName="MetricVal")
        hr_box.addWidget(self.lbl_pred_hr)

        ss_box = QVBoxLayout()
        ss_box.addWidget(QLabel("t+1 睡眠分数", objectName="CardTitle"))
        self.lbl_pred_ss = QLabel("--", objectName="MetricVal")
        ss_box.addWidget(self.lbl_pred_ss)

        pred_layout.addLayout(hr_box)
        pred_layout.addLayout(ss_box)
        cards_layout.addWidget(pred_card, 2)

        error_card = create_glass_card()
        error_layout = QHBoxLayout(error_card)
        self.lbl_rmse = QLabel("--", objectName="ErrorVal")
        self.lbl_mae = QLabel("--", objectName="ErrorVal")
        self.lbl_r2 = QLabel("--", objectName="ErrorVal")

        for name, lbl in [("全局 RMSE", self.lbl_rmse), ("平均绝对误差 MAE", self.lbl_mae),
                          ("决定系数 R²", self.lbl_r2)]:
            vbox = QVBoxLayout()
            vbox.addWidget(QLabel(name, objectName="CardTitle"))
            vbox.addWidget(lbl)
            error_layout.addLayout(vbox)

        cards_layout.addWidget(error_card, 3)
        right_layout.addLayout(cards_layout)

        chart_card = create_glass_card()
        chart_layout = QVBoxLayout(chart_card)
        self.canvas = AcademicDarkCanvas(self, width=7, height=5, dpi=100)
        chart_layout.addWidget(self.canvas)
        right_layout.addWidget(chart_card)

        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([320, 1000])
        self.splitter.setStyleSheet("QSplitter::handle { background-color: transparent; }")

        main_layout.addWidget(self.splitter)

        # 右下角窗口缩放把手
        grip_layout = QHBoxLayout()
        grip_layout.setContentsMargins(0, 0, 0, 0)
        grip_layout.addStretch()
        size_grip = QSizeGrip(self)
        size_grip.setStyleSheet("background-color: transparent;")
        grip_layout.addWidget(size_grip)
        main_layout.addLayout(grip_layout)

    def toggle_sidebar(self):
        if self.left_panel.isVisible():
            self.left_panel.hide()
            self.btn_toggle.setText("▶ 展开控制面板")
        else:
            self.left_panel.show()
            self.btn_toggle.setText("◀ 收起控制面板")

    def create_slider(self, layout, name, min_v, max_v, default_v, divisor):
        hbox = QHBoxLayout()
        lbl_name = QLabel(name)
        lbl_name.setStyleSheet("background: transparent;")

        spin_box = QDoubleSpinBox()
        spin_box.setRange(min_v / divisor, max_v / divisor)
        spin_box.setSingleStep(1.0 / divisor if divisor > 1 else 1.0)
        spin_box.setDecimals(2 if divisor > 1 else 0)
        spin_box.setValue(default_v / divisor)
        spin_box.setFixedWidth(55)
        spin_box.setStyleSheet("""
            QDoubleSpinBox {
                color: #FFFFFF; font-weight: bold; background: rgba(0, 0, 0, 0.3);
                border: 1px solid transparent; border-radius: 3px;
            }
            QDoubleSpinBox:hover { border: 1px solid #333333; }
            QDoubleSpinBox:focus { border: 1px solid #888888; background: #000000;}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; } 
        """)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default_v)

        slider.valueChanged.connect(lambda val: spin_box.setValue(val / divisor))
        spin_box.valueChanged.connect(lambda val: slider.setValue(int(val * divisor)))

        hbox.addWidget(lbl_name)
        hbox.addWidget(slider)
        hbox.addWidget(spin_box)
        layout.addLayout(hbox)

        return slider, divisor

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "导入生理数据", "", "CSV Files (*.csv)")
        if file_name:
            try:
                df = pd.read_csv(file_name)
                if 'HeartRate' not in df.columns or 'SleepScore' not in df.columns:
                    QMessageBox.critical(self, "错误", "CSV 表头必须包含 'HeartRate' 和 'SleepScore'。")
                    return
                self.df_user = df
                self.lbl_file_status.setText(f"已加载 ({len(df)} 组数据)")
                self.lbl_file_status.setStyleSheet("color: #FFFFFF;")
            except Exception as e:
                QMessageBox.critical(self, "读取失败", str(e))

    def parse_manual_input(self):
        hr_str = self.text_hr.toPlainText().strip()
        ss_str = self.text_ss.toPlainText().strip()
        if not hr_str or not ss_str: raise ValueError("快捷输入框不能为空。")
        hr_list = [float(x) for x in re.split(r'[,\s]+', hr_str) if x]
        ss_list = [float(x) for x in re.split(r'[,\s]+', ss_str) if x]
        if len(hr_list) < 10 or len(ss_list) < 10: raise ValueError("至少需要 10 天的数据。")
        return hr_list, ss_list

    def save_current_data(self):
        """将当前用户输入或载入的数据安全地持久化到本地"""
        try:
            # 1. 动态获取当前激活面板的数据
            if self.tabs.currentIndex() == 0:
                hr_list, ss_list = self.parse_manual_input()
            else:
                if self.df_user is None:
                    raise ValueError("当前没有加载任何 CSV 数据。")
                hr_list = self.df_user['HeartRate'].tolist()
                ss_list = self.df_user['SleepScore'].tolist()

            # 2. 内存安全的数据重组：对齐长度，利用 Pandas 内部的 C 引擎处理
            min_len = min(len(hr_list), len(ss_list))

            # 局部变量 export_df，使用完毕后会被 Python GC 自动回收，绝无内存泄漏风险
            export_df = pd.DataFrame({
                'HeartRate': hr_list[:min_len],
                'SleepScore': ss_list[:min_len]
            })

            # 3. 呼出保存对话框
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出/保存生理数据",
                "ESN_Physio_Data.csv",
                "CSV Files (*.csv)",
                options=options
            )

            if file_path:

                export_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "导出成功", f"数据已安全保存至:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "无法保存", str(e))



    def run_prediction(self):
        self.btn_run.setText("演算中...")
        self.btn_run.setEnabled(False)
        QApplication.processEvents()

        try:
            # 1. 数据获取
            if self.tabs.currentIndex() == 0:
                hr_list, ss_list = self.parse_manual_input()
            else:
                if self.df_user is None: raise ValueError("请先载入 CSV 文件。")
                hr_list = self.df_user['HeartRate'].tolist()
                ss_list = self.df_user['SleepScore'].tolist()

            pipeline = DataPipeline(hr_list, ss_list, test_size=0.2)
            dataset = pipeline.process_data()

            # 获取滑块参数
            u = self.slider_units.value()
            sr = self.slider_sr.value() / self.val_sr
            lr = self.slider_lr.value() / self.val_lr
            rc = self.slider_sparsity.value() / self.val_sparsity
            rd = self.slider_ridge.value() / self.val_ridge


            rpy.set_seed(42)
            eval_model = ESNPredictor(units=u, sr=sr, lr=lr, rc_connectivity=rc, ridge=rd)
            eval_model.train(dataset['X_train'], dataset['Y_train'], warmup=10)

            # 在测试集上进行预测
            test_pred_scaled = eval_model.predict(dataset['X_test'])
            test_pred = pipeline.inverse_transform(test_pred_scaled)
            test_real = pipeline.inverse_transform(dataset['Y_test'])

            # 修正 2：分离量纲，独立计算心率与睡眠的误差
            rmse_hr = np.sqrt(mean_squared_error(test_real[:, 0], test_pred[:, 0]))
            rmse_ss = np.sqrt(mean_squared_error(test_real[:, 1], test_pred[:, 1]))

            mae_hr = mean_absolute_error(test_real[:, 0], test_pred[:, 0])
            mae_ss = mean_absolute_error(test_real[:, 1], test_pred[:, 1])

            r2_hr = r2_score(test_real[:, 0], test_pred[:, 0])
            r2_ss = r2_score(test_real[:, 1], test_pred[:, 1])


            deploy_model = ESNPredictor(units=u, sr=sr, lr=lr, rc_connectivity=rc, ridge=rd)
            deploy_model.train(dataset['X_all'], dataset['Y_all'], warmup=10)


            tomorrow_pred_scaled = deploy_model.predict_single_step(dataset['last_today_input'])
            tomorrow_final = pipeline.inverse_transform(tomorrow_pred_scaled.reshape(1, -1))


            all_predictions = pipeline.inverse_transform(deploy_model.predict(dataset['X_all']))
            real_data = pipeline.inverse_transform(dataset['Y_all'])


            self.lbl_pred_hr.setText(f"{tomorrow_final[0, 0]:.1f}")
            self.lbl_pred_ss.setText(f"{tomorrow_final[0, 1]:.1f}")

            # 将 UI 上的标签更新为分离展示的形式
            self.lbl_rmse.setText(f"HR: {rmse_hr:.2f}  |  Sleep: {rmse_ss:.2f}")
            self.lbl_mae.setText(f"HR: {mae_hr:.2f}  |  Sleep: {mae_ss:.2f}")
            self.lbl_r2.setText(f"HR: {r2_hr:.2f}  |  Sleep: {r2_ss:.2f}")

            # 绘制图表
            self.plot_academic_results(real_data, all_predictions)

        except Exception as e:
            QMessageBox.critical(self, "中断", str(e))
        finally:
            self.btn_run.setText("启动推演")
            self.btn_run.setEnabled(True)

    def plot_academic_results(self, real_data, pred_data):
        ax1 = self.canvas.ax1
        ax2 = self.canvas.ax2
        ax1.clear()
        ax2.clear()

        DISPLAY_LIMIT = 50
        total_len = len(real_data)

        if total_len > DISPLAY_LIMIT:
            plot_real = real_data[-DISPLAY_LIMIT:]
            plot_pred = pred_data[-DISPLAY_LIMIT:]
            time_steps = np.arange(total_len - DISPLAY_LIMIT, total_len)
        else:
            plot_real = real_data
            plot_pred = pred_data
            time_steps = np.arange(total_len)

        ax1.plot(time_steps, plot_real[:, 0], color='#666666', linestyle='-', linewidth=1.5,
                 marker='o', markersize=4, markerfacecolor='#121212', label='Actual HR')
        ax1.plot(time_steps, plot_pred[:, 0], color='#FFFFFF', linestyle='--', linewidth=2,
                 marker='s', markersize=4, label='Predicted HR')

        ax1.set_xlabel('Time Steps (Days)', fontdict=self.canvas.label_font, labelpad=10)
        ax1.set_ylabel('Resting Heart Rate', fontdict=self.canvas.label_font, color='#FFFFFF', labelpad=10)
        ax1.tick_params(axis='y', colors='#FFFFFF')

        ax2.plot(time_steps, plot_real[:, 1], color='#333333', linestyle='-', linewidth=1.5,
                 marker='^', markersize=4, markerfacecolor='#121212', label='Actual Sleep')
        ax2.plot(time_steps, plot_pred[:, 1], color='#AAAAAA', linestyle='-.', linewidth=2,
                 marker='D', markersize=4, label='Predicted Sleep')

        ax2.set_ylabel('Sleep Quality Score', fontdict=self.canvas.label_font, color='#AAAAAA', rotation=270,
                       labelpad=15)
        ax2.tick_params(axis='y', colors='#AAAAAA')

        ax1.grid(True, linestyle='-', alpha=0.1, color='#FFFFFF')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center',
                   bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False, labelcolor='#AAAAAA', fontsize=9)

        y1_max, y1_min = max(plot_real[:, 0].max(), plot_pred[:, 0].max()), min(plot_real[:, 0].min(),
                                                                                plot_pred[:, 0].min())
        ax1.set_ylim(y1_min - 3, y1_max + (y1_max - y1_min) * 0.35)

        y2_max, y2_min = max(plot_real[:, 1].max(), plot_pred[:, 1].max()), min(plot_real[:, 1].min(),
                                                                                plot_pred[:, 1].min())
        ax2.set_ylim(y2_min - 5, y2_max + (y2_max - y2_min) * 0.35)

        self.canvas.draw()


    def resizeEvent(self, event):
        super().resizeEvent(event)

        # 获取当前窗口高度，并以 850px 为基准计算放大比例
        current_height = self.height()
        scale = current_height / 600.0

        if scale < 1.0:
            scale = 1.0  # 保证字体在缩小窗口时不会变得太小无法看清

        # 动态计算新的字号 (基础字号 * 放大比例)
        val_size = int(28 * scale)
        err_size = int(18 * scale)
        title_size = int(11 * scale)

        # 构建覆盖原有固定字号的动态 QSS
        dynamic_qss = f"""
            QLabel#MetricVal {{ font-size: {val_size}px; font-weight: 900; background: transparent; }}
            QLabel#ErrorVal {{ font-size: {err_size}px; font-weight: bold; color: #FFFFFF; background: transparent; }}
            QLabel#CardTitle {{ font-size: {title_size}px; color: #777777; background: transparent; }}
        """

        # 获取原有的全局样式表，并将动态样式追加到末尾进行覆盖
        current_style = self.styleSheet()
        # 清除之前可能追加过的旧动态样式，避免样式表无限变长
        base_style = current_style.split("/* DYNAMIC_FONTS */")[0]

        self.setStyleSheet(base_style + "\n/* DYNAMIC_FONTS */\n" + dynamic_qss)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = ESNDesktopApp()
    window.show()
    sys.exit(app.exec_())
