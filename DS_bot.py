from openai import OpenAI
import os
import time


class DS_Bot:
    def __init__(self, api_key=None, model="deepseek-chat", temperature=0.7, max_tokens=1000):
        """
        初始化基于Deepseek API的聊天机器人

        参数:
            api_key: Deepseek API密钥（默认从环境变量获取）
            model: 使用的Deepseek模型
            temperature: 响应的采样温度
            max_tokens: 响应的最大token数
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥必须提供或设置为DEEPSEEK_API_KEY环境变量")

        # Deepseek API使用OpenAI的客户端与基本URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"  # Deepseek API端点
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history = []

    def add_message(self, role, content):
        """向对话历史添加消息"""
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_input, stream=False):
        """获取Deepseek API对用户输入的响应"""
        # 检查是否是命令
        if user_input.startswith("/"):
            return self.handle_command(user_input)

        # 添加用户消息到历史记录
        self.add_message("user", user_input)

        try:
            # 调用Deepseek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )

            if stream:
                return self._handle_streaming(response)
            else:
                assistant_message = response.choices[0].message.content
                self.add_message("assistant", assistant_message)
                return assistant_message

        except Exception as e:
            error_msg = f"调用Deepseek API时出错: {str(e)}"
            print(error_msg)
            return error_msg

    def handle_command(self, command):
        cmd = command.lower().strip()

        # 命令集
        if cmd == "/help":
            return ("可用命令:\n"
                    "/help - 显示帮助信息\n"
                    "/clear - 清除对话历史\n"
                    "/restart - 重新开始对话\n"
                    "/mode - 显示当前模式\n"
                    "/model - 显示当前使用的模型\n"
                    "/image - 生成图像描述（仅高级模式）")
        elif cmd == "/clear" or cmd == "/restart":
            self.clear_history()
            return "对话历史已清除。"
        elif cmd == "/mode":
            return "当前使用的是高级模式，拥有完整的AI功能。"
        elif cmd == "/model":
            return f"当前使用的模型: {self.model}"
        elif cmd.startswith("/image"):
            try:
                # 简单的图像描述生成
                prompt = command[7:].strip() or "一个美丽的风景"
                return f"[图像生成] 基于提示词: '{prompt}'"
            except Exception as e:
                return f"图像生成失败: {str(e)}"
        else:
            return f"未知命令: {command}。输入 /help 获取可用命令列表。"

    def _handle_streaming(self, response_stream):
        """处理流式响应"""
        collected_content = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                collected_content += content_chunk
                print(content_chunk, end="", flush=True)

        print()  # 最后的换行
        self.add_message("assistant", collected_content)
        return collected_content

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []

    def run_interactive(self):
        """运行交互式控制台会话"""
        print("Deepseek机器人已初始化。输入'exit'结束对话。")

        while True:
            user_input = input("\n您: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("机器人: 再见！")
                break

            print("\n机器人: ", end="", flush=True)
            self.get_response(user_input, stream=True)


# 使用示例
if __name__ == "__main__":
    bot = DS_Bot()  # 在这里提供API密钥或设置环境变量
    bot.run_interactive()