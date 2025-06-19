#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: llm_agent.py
@time: 2025/6/19 21:12
@project: my-team-learning
@desc: LLM Agent的实现
"""
import inspect
import os
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

loaded = load_dotenv(find_dotenv(), override=True)

API_KEY = os.getenv("SiliconFlow_API_KEY")
BASE_URL = os.getenv("SiliconFlow_BASE_URL")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def add(a: float, b: float):
    """
    计算两个浮点数的和。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的和。
    """
    return str(a + b)


def mul(a: float, b: float):
    """
    计算两个浮点数的积。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的积。
    """
    return str(a * b)


def compare(a: float, b: float):
    """
    比较两个浮点数的大小。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 比较结果的字符串表示。
    """
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'


def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    string = a.lower()
    letter = b.lower()

    count = string.count(letter)
    return f"The letter '{letter}' appears {count} times in the string."


def function_to_json(func) -> dict:
    # 定义 Python 类型到 JSON 数据类型的映射
    type_map = {
        str: "string",  # 字符串类型映射为 JSON 的 "string"
        int: "integer",  # 整型类型映射为 JSON 的 "integer"
        float: "number",  # 浮点型映射为 JSON 的 "number"
        bool: "boolean",  # 布尔型映射为 JSON 的 "boolean"
        list: "array",  # 列表类型映射为 JSON 的 "array"
        dict: "object",  # 字典类型映射为 JSON 的 "object"
        type(None): "null",  # None 类型映射为 JSON 的 "null"
    }

    # 获取函数的签名信息
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        # 如果获取签名失败，则抛出异常并显示具体的错误信息
        raise ValueError(
            f"无法获取函数 {func.__name__} 的签名: {str(e)}"
        )

    # 用于存储参数信息的字典
    parameters = {}
    for param in signature.parameters.values():
        # 尝试获取参数的类型，如果无法找到对应的类型则默认设置为 "string"
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            # 如果参数类型不在 type_map 中，抛出异常并显示具体错误信息
            raise KeyError(
                f"未知的类型注解 {param.annotation}，参数名为 {param.name}: {str(e)}"
            )
        # 将参数名及其类型信息添加到参数字典中
        parameters[param.name] = {"type": param_type}

    # 获取函数中所有必需的参数（即没有默认值的参数）
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # 返回包含函数描述信息的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,  # 函数的名称
            "description": func.__doc__ or "",  # 函数的文档字符串（如果不存在则为空字符串）
            "parameters": {
                "type": "object",
                "properties": parameters,  # 函数参数的类型描述
                "required": required,  # 必须参数的列表
            },
        },
    }


class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools=None,
                 verbose: bool = True):
        if tools is None:
            tools = []
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # 处理工具调用
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content


if __name__ == "__main__":
    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
    )

    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
