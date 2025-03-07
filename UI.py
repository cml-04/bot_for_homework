import sys
import os
import json
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLineEdit, QTextEdit, QLabel, QComboBox,
                           QTabWidget, QSplitter, QDialog, QFileDialog, QMessageBox,
                           QFormLayout, QGroupBox, QCheckBox, QStackedWidget, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSlot,QTimer
from PyQt5.QtGui import QFont, QIcon
from DS_bot import DS_Bot
from simple_bot import SimpleBot
from database import UserDatabase


class MessageInput(QTextEdit):
    """自定义文本输入框，Enter键发送消息，Ctrl+Enter键换行"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def keyPressEvent(self, event):
        # 检查是否按下Enter键
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # 检查是否同时按下Ctrl键
            if event.modifiers() & Qt.ControlModifier:
                # Ctrl+Enter添加换行符
                super().keyPressEvent(event)
            else:
                # 仅Enter键则发送消息
                if self.parent:
                    self.parent.send_message()
        else:
            # 其他键正常处理
            super().keyPressEvent(event)

class LoginDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.user_data = None
        self.setWindowTitle("登录")
        self.resize(350, 200)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Username field
        self.username_input = QLineEdit()
        form_layout.addRow("用户名:", self.username_input)

        # Password field
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)

        layout.addLayout(form_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        self.register_btn = QPushButton("注册")
        self.register_btn.clicked.connect(self.open_register)

        self.forgot_btn = QPushButton("忘记密码")
        self.forgot_btn.clicked.connect(self.open_forgot_password)

        btn_layout.addWidget(self.login_btn)
        btn_layout.addWidget(self.register_btn)
        btn_layout.addWidget(self.forgot_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "错误", "请输入用户名和密码")
            return

        user_data = self.db.authenticate(username, password)
        if user_data:
            self.user_data = user_data
            self.accept()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码不正确")

    def open_register(self):
        dialog = RegisterDialog(self.db, self)
        if dialog.exec_() == QDialog.Accepted:
            self.username_input.setText(dialog.username)
            QMessageBox.information(self, "注册成功", "账号创建成功，请使用新账号登录")

    def open_forgot_password(self):
        dialog = ForgotPasswordDialog(self.db, self)
        dialog.exec_()


class RegisterDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.username = ""
        self.setWindowTitle("注册新账号")
        self.resize(400, 250)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Username field
        self.username_input = QLineEdit()
        form_layout.addRow("用户名:", self.username_input)

        # Email field
        self.email_input = QLineEdit()
        form_layout.addRow("电子邮箱:", self.email_input)

        # Password field
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)

        # Confirm password field
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("确认密码:", self.confirm_password_input)

        # API Key field
        self.api_key_input = QLineEdit()
        form_layout.addRow("Deepseek API密钥 (可选):", self.api_key_input)

        layout.addLayout(form_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.register_btn = QPushButton("注册")
        self.register_btn.clicked.connect(self.register)
        self.register_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.register_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def register(self):
        username = self.username_input.text().strip()
        email = self.email_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()
        api_key = self.api_key_input.text().strip()

        # Validation
        if not username or not email or not password:
            QMessageBox.warning(self, "错误", "请填写所有必填字段")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致")
            return

        # Email validation (simple check)
        if "@" not in email or "." not in email:
            QMessageBox.warning(self, "错误", "请输入有效的电子邮箱地址")
            return

        # Register the user
        success = self.db.register_user(username, password, email, api_key)
        if success:
            self.username = username
            self.accept()
        else:
            QMessageBox.warning(self, "注册失败", "用户名或电子邮箱已被使用")


class ForgotPasswordDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("找回密码")
        self.resize(350, 200)
        self.setup_ui()

    def setup_ui(self):
        self.stack = QStackedWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)

        # Email input page
        email_widget = QWidget()
        email_layout = QVBoxLayout(email_widget)

        email_layout.addWidget(QLabel("请输入您的注册邮箱:"))
        self.email_input = QLineEdit()
        email_layout.addWidget(self.email_input)

        submit_btn = QPushButton("提交")
        submit_btn.clicked.connect(self.send_reset_token)
        email_layout.addWidget(submit_btn)

        # Reset token page
        token_widget = QWidget()
        token_layout = QVBoxLayout(token_widget)

        token_layout.addWidget(QLabel("请输入重置令牌:"))
        self.token_input = QLineEdit()
        token_layout.addWidget(self.token_input)

        token_layout.addWidget(QLabel("新密码:"))
        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        token_layout.addWidget(self.new_password)

        token_layout.addWidget(QLabel("确认新密码:"))
        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        token_layout.addWidget(self.confirm_password)

        reset_btn = QPushButton("重置密码")
        reset_btn.clicked.connect(self.reset_password)
        token_layout.addWidget(reset_btn)

        self.stack.addWidget(email_widget)
        self.stack.addWidget(token_widget)
        self.setLayout(main_layout)

    def send_reset_token(self):
        email = self.email_input.text().strip()
        if not email:
            QMessageBox.warning(self, "错误", "请输入电子邮箱")
            return

        token = self.db.generate_reset_token(email)
        if token:
            # In a real application, you would send this token via email
            QMessageBox.information(self, "重置令牌", f"您的重置令牌是: {token}\n"
                                                      "（在实际应用中，这将通过电子邮件发送）")
            self.stack.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "错误", "未找到该电子邮箱")

    def reset_password(self):
        token = self.token_input.text().strip()
        new_password = self.new_password.text()
        confirm = self.confirm_password.text()

        if not token or not new_password:
            QMessageBox.warning(self, "错误", "请填写所有字段")
            return

        if new_password != confirm:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致")
            return

        success = self.db.reset_password(token, new_password)
        if success:
            QMessageBox.information(self, "成功", "密码已成功重置")
            self.accept()
        else:
            QMessageBox.warning(self, "错误", "无效的重置令牌或令牌已过期")


class ChatTab(QWidget):
    def __init__(self, parent=None, api_key="", title="新对话", use_advanced=True):
        """单个聊天标签页"""
        super().__init__(parent)
        self.title = title
        self.api_key = api_key
        self.bot = None
        self.use_advanced = use_advanced

        # 创建机器人实例
        if use_advanced and api_key:
            try:
                self.bot = DS_Bot(api_key=self.api_key)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"初始化高级机器人时出错: {str(e)}")
                self.bot = SimpleBot()
        else:
            self.bot = SimpleBot()

        # 创建UI
        self.init_ui()

    def init_ui(self):
        """初始化聊天界面"""
        layout = QVBoxLayout()

        # 状态指示器
        status_layout = QHBoxLayout()
        if self.use_advanced:
            status_label = QLabel("高级模式 ✓")
            status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            status_label = QLabel("简易模式 ⚠")
            status_label.setStyleSheet("color: orange; font-weight: bold;")
        status_layout.addWidget(status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # 聊天历史区域
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setAcceptRichText(True)
        self.chat_history.setStyleSheet("background-color: #f5f5f5; border-radius: 5px;")
        layout.addWidget(self.chat_history)

        # 输入区域
        input_layout = QHBoxLayout()
        # 使用自定义的MessageInput类，并传入self作为parent
        self.message_input = MessageInput(self)
        self.message_input.setPlaceholderText("输入消息...")
        self.message_input.setMaximumHeight(100)
        self.message_input.setStyleSheet("border-radius: 5px;")
        input_layout.addWidget(self.message_input, 4)

        # 发送按钮
        send_button = QPushButton("发送")
        send_button.setMinimumHeight(40)
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        input_layout.addWidget(send_button, 1)

        layout.addLayout(input_layout)
        self.setLayout(layout)

        # 添加欢迎消息
        if not self.use_advanced:
            self.chat_history.append(
                "<b>系统提示:</b> 您正在使用简易模式。如需使用高级功能，请确保提供有效的API密钥并安装GPU支持。")

    def send_message(self):
        """发送消息并获取回复"""
        if not self.bot:
            QMessageBox.warning(self, "错误", "机器人未初始化")
            return

        message = self.message_input.toPlainText().strip()
        if not message:
            return

        # 显示用户消息
        self.chat_history.append(f"<div style='text-align: right;'><b>您:</b> {message}</div>")
        self.message_input.clear()

        # 获取并显示机器人回复
        self.chat_history.append("<b>机器人:</b> <i>思考中...</i>")
        QApplication.processEvents()  # 更新UI

        try:
            if isinstance(self.bot, DS_Bot):
                response = self.bot.get_response(message)
            else:
                response = self.bot.get_response(message)

            # 更新最后一行，删除"思考中"并添加回复
            cursor = self.chat_history.textCursor()
            cursor.movePosition(cursor.End)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # 删除额外的换行符
            self.chat_history.append(f"<b>机器人:</b> {response}")

        except Exception as e:
            self.chat_history.append(f"<b>错误:</b> {str(e)}")

        # 滚动到底部
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum())

    def update_api_key(self, api_key, use_advanced=True):
        """更新API密钥和模式"""
        self.api_key = api_key
        self.use_advanced = use_advanced

        try:
            if use_advanced and api_key:
                self.bot = DS_Bot(api_key=self.api_key)
            else:
                self.bot = SimpleBot()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"更新机器人时出错: {str(e)}")
            self.bot = SimpleBot()


class ChatBotUI(QMainWindow):
    def __init__(self):
        """主应用窗口"""
        super().__init__()
        self.db = UserDatabase()
        self.api_key = ""
        self.username = ""
        self.use_advanced = False
        self.check_login()

    def change_bot_type(self, index):
        """Change the bot type based on tab selection"""
        self.current_bot_type = "simple" if index == 0 else "advanced"

        # If advanced selected but not available, show warning
        if self.current_bot_type == "advanced" and not self.use_advanced:
            QMessageBox.warning(
                self,
                "功能受限",
                "高级模式需要API密钥和GPU支持。请在设置中配置API密钥。"
            )

    def create_new_chat(self):
        """Create a new chat tab"""
        count = self.tab_widget.count()

        # Use the currently selected bot type
        use_advanced = self.current_bot_type == "advanced" and self.use_advanced

        chat_tab = ChatTab(
            api_key=self.api_key,
            title=f"对话 {count + 1}",
            use_advanced=use_advanced
        )

        # Set tab icon based on bot type
        icon = QIcon("icons/advanced_bot.png" if use_advanced else "icons/simple_bot.png")

        self.tab_widget.addTab(chat_tab, icon, f"对话 {count + 1}")
        self.tab_widget.setCurrentIndex(count)

    def check_login(self):
        """检查用户登录状态"""
        dialog = LoginDialog(self.db)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            self.user_data = dialog.user_data
            self.username = dialog.user_data["username"]
            self.api_key = dialog.user_data["api_key"]

            # 检查GPU和API密钥
            self.check_requirements()
            self.init_ui()
        else:
            # 用户取消登录，退出应用
            sys.exit(0)

    def check_requirements(self):
        """检查高级模式所需的条件"""
        gpu_available = torch.cuda.is_available()
        api_valid = bool(self.api_key)

        self.use_advanced = gpu_available and api_valid

        if not self.use_advanced:
            warnings = []
            if not gpu_available:
                warnings.append("- 未检测到GPU支持")
            if not api_valid:
                warnings.append("- 未设置API密钥")

            QMessageBox.warning(
                self,
                "功能受限",
                f"由于以下原因，您将使用简易模式:\n{''.join(warnings)}\n\n如需使用高级功能，请确保满足上述条件。"
            )

        # 如果新注册用户没有API密钥，提示设置，但延迟到UI初始化之后
        self.needs_api_setup = not api_valid

    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle(f"AI聊天助手 - {self.username}")
        self.setGeometry(100, 100, 900, 600)

        # Set application icon
        self.setWindowIcon(QIcon("icons/chat_icon.png"))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create a splitter for left sidebar and main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left sidebar for bot selection
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Bot selection label
        select_label = QLabel("选择机器人")
        select_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(select_label)

        # Bot selection tabs (vertical)
        self.bot_tabs = QTabWidget()
        self.bot_tabs.setTabPosition(QTabWidget.West)  # Tabs on left side

        # Simple bot tab
        simple_bot_widget = QWidget()
        simple_bot_layout = QVBoxLayout(simple_bot_widget)
        simple_bot_info = QLabel("简易机器人\n\n• 本地运行\n• 无需API\n• 基础对话功能")
        simple_bot_info.setWordWrap(True)
        simple_bot_layout.addWidget(simple_bot_info)
        simple_bot_layout.addStretch()
        self.bot_tabs.addTab(simple_bot_widget, "简易")

        # Advanced bot tab
        ds_bot_widget = QWidget()
        ds_bot_layout = QVBoxLayout(ds_bot_widget)
        ds_bot_info = QLabel("高级AI\n\n• DeepSeek AI\n• 需要API密钥\n• 高级对话理解\n• 图像生成")
        ds_bot_info.setWordWrap(True)
        ds_bot_layout.addWidget(ds_bot_info)
        ds_bot_layout.addStretch()
        self.bot_tabs.addTab(ds_bot_widget, "高级")

        # Connect bot selection signal
        self.bot_tabs.currentChanged.connect(self.change_bot_type)

        left_layout.addWidget(self.bot_tabs)

        # API settings button
        api_btn = QPushButton("API设置")
        api_btn.clicked.connect(self.show_api_settings)
        left_layout.addWidget(api_btn)

        left_layout.addStretch()

        # Right content area
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Chat tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        right_layout.addWidget(self.tab_widget)

        # Bottom buttons
        button_layout = QHBoxLayout()

        # New chat button
        new_chat_btn = QPushButton("新对话")
        new_chat_btn.clicked.connect(self.create_new_chat)
        new_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 8px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        button_layout.addWidget(new_chat_btn)

        button_layout.addStretch()

        # User info
        user_info = QLabel(f"用户: {self.username}")
        button_layout.addWidget(user_info)

        # Logout button
        logout_btn = QPushButton("注销")
        logout_btn.clicked.connect(self.logout)
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #607d8b;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #455a64;
            }
        """)
        button_layout.addWidget(logout_btn)

        right_layout.addLayout(button_layout)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 700])  # Set initial sizes

        # Create initial chat
        self.current_bot_type = "simple" if not self.use_advanced else "advanced"
        self.create_new_chat()

        # Prompt for API setup if needed
        if hasattr(self, 'needs_api_setup') and self.needs_api_setup:
            QTimer.singleShot(100, self.prompt_api_settings)

    def prompt_api_settings(self):
        """提示用户设置API密钥"""
        response = QMessageBox.question(
            self,
            "设置API密钥",
            "您尚未设置Deepseek API密钥，是否现在设置？",
            QMessageBox.Yes | QMessageBox.No
        )

        if response == QMessageBox.Yes:
            self.show_api_settings()

    def create_new_chat(self):
        """创建新的对话标签页"""
        count = self.tab_widget.count()
        chat_tab = ChatTab(
            api_key=self.api_key,
            title=f"对话 {count + 1}",
            use_advanced=self.use_advanced
        )
        self.tab_widget.addTab(chat_tab, f"对话 {count + 1}")
        self.tab_widget.setCurrentIndex(count)

    def close_tab(self, index):
        """关闭对话标签页"""
        if self.tab_widget.count() > 1:
            self.tab_widget.removeTab(index)
        else:
            # 至少保留一个对话
            QMessageBox.information(self, "提示", "至少需要保留一个对话")

    def show_api_settings(self):
        """显示API设置对话框"""
        api_key, ok = QInputDialog.getText(
            self, "API设置",
            "请输入Deepseek API密钥:",
            QLineEdit.Normal,
            self.api_key
        )

        if ok:
            self.api_key = api_key.strip()

            # 更新数据库中的API密钥
            self.db.update_api_key(self.username, self.api_key)

            # 重新检查需求
            self.check_requirements()

            # 更新所有标签页
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                tab.update_api_key(self.api_key, self.use_advanced)

            QMessageBox.information(self, "成功", "API设置已更新")

    def logout(self):
        """注销当前用户"""
        reply = QMessageBox.question(
            self, '确认注销',
            '确定要注销当前账号吗?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 重启应用程序逻辑
            QApplication.quit()
            program = sys.executable
            os.execl(program, program, *sys.argv)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用更现代的风格

    window = ChatBotUI()
    window.show()

    sys.exit(app.exec_())