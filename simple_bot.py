# simple_bot.py
import re
import random


class SimpleBot:
    def __init__(self):
        self.patterns = {
            r'hello|hi|hey': ['Hello!', 'Hi there!', 'Hey! How can I help you?'],
            r'how are you': ['I\'m doing well, thanks!', 'I\'m a simple assistant, ready to help.'],
            r'bye|goodbye': ['Goodbye!', 'See you later!', 'Have a great day!'],
            r'help': [
                'I\'m a simple assistant with limited functionality. For advanced features, please provide a valid API key and ensure GPU support.'],
            r'api|key|deepseek': [
                'To use advanced features, you need to provide a valid Deepseek API key in your profile settings.'],
            r'gpu|cuda': ['GPU support is required for advanced features. Please install necessary drivers.'],
            r'login|account|register': ['You can manage your account from the login screen.']
        }
        self.default_responses = [
            "I'm a simple assistant with limited functionality. For advanced features, please provide a valid API key and ensure GPU support.",
            "I understand your message, but I have limited capabilities. Advanced features require API key and GPU support.",
            "As a basic assistant, I can only provide simple responses. Please upgrade for more capabilities."
        ]

    def get_response(self, user_input):
        # Check if this is a command (starts with /)
        if user_input.startswith("/"):
            return self.handle_command(user_input)

        user_input = user_input.lower()

        for pattern, responses in self.patterns.items():
            if re.search(pattern, user_input):
                return random.choice(responses)

        return random.choice(self.default_responses)

    def handle_command(self, command):
        cmd = command.lower().strip()

        # Command set
        if cmd == "/help":
            return ("可用命令:\n"
                    "/help - 显示帮助信息\n"
                    "/clear - 清除对话历史\n"
                    "/restart - 重新开始对话\n"
                    "/mode - 显示当前模式")
        elif cmd == "/clear" or cmd == "/restart":
            return "对话历史已清除。"
        elif cmd == "/mode":
            return "当前使用的是简易模式，功能有限。"
        else:
            return f"未知命令: {command}。输入 /help 获取可用命令列表。"