# Task03 单智能体开发

## 1 Agent简介

- Agent的组成：Agent = 大语言模型（LLM） + 观察 + 思考 + 行动 + 记忆

- 观察：智能体的感知机制，感知另一个智能体的文本消息、视觉数据或者来自客户服务的音频信息等一系列信号。
- 思考：智能体内部的决策过程，思考设计分析观察结果和记忆内容并考虑可能的行动。
- 行动：智能体对思考和观察的显式响应。
- 记忆：智能体的记忆存储过去的经验，相当于历史记录。

## 2 实现一个单动作Agent

- 背景介绍：使用MetaGPT框架，实现一个生成代码的Agent，根据用户需求生成代码。
- 需求分析：实现一个`SimpleCoder`，获取用户输入的需求，记忆用户需求，编写对应的代码。

### 2.1 编写SimpleWriteCode动作


```python
import re
import asyncio
from metagpt.actions import Action
```


```python
class SimpleWriteCode(Action):
    
    # 如果使用Zhipu，需要修改prompt
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python``` with NO other texts, 
    """
    
    # 动作的指定名称
    name: str = "SimpleWriteCode"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        rsp = await self._aask(prompt)
        code_text = SimpleWriteCode.parse_code(rsp)
        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
```

### 2.2 设计SimpleCoder角色


```python
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger
```


```python
class SimpleCoder(Role):
    
    # 昵称
    name: str = "Alice"
    # 人设    
    profile: str = "SimpleCoder"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始Action，SimpleWriteCode加入到todo中
        self._init_actions([SimpleWriteCode])
    
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        todo = self.rc.todo # todo will be SimpleWriteCode()
        
        # 获取最近的一条记忆，即用户的需求
        msg = self.get_memories(k=1)[0] # find the most recent messages
        
        # 将需求的内容传递给Action，即instruction
        code_text = await todo.run(msg.content)
        # 返回封装的Message对象
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg
```

### 2.3 运行SimpleCoder角色


```python
async def main():
    msg = "write a function that calculates the sum of a list"
    role = SimpleCoder()
    result = await role.run(msg)
    logger.info(result)
```


```python
await main()
```

    2024-02-29 16:08:38.907 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-02-29 16:08:38.909 | INFO     | __main__:_act:14 - Alice(SimpleCoder): ready to SimpleWriteCode
    

     ```python
    def sum_of_list(lst):
        return sum(lst)
    
    # Test Case 1
    test_list1 = [1, 2, 3, 4, 5]
    result1 = sum_of_list(test_list1)
    print(f"The sum of the list {test_list1} is {result1}")
    
    # Test Case 2
    test_list2 = [10, -5, 7, 3, 20]
    result2 = sum_of_list(test_list2)
    print(f"The sum of the list {test_list2} is {result2}")
    ```

    2024-02-29 16:08:47.276 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.002 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 43, completion_tokens: 157
    2024-02-29 16:08:47.277 | INFO     | __main__:main:5 - SimpleCoder: 
    def sum_of_list(lst):
        return sum(lst)
    
    # Test Case 1
    test_list1 = [1, 2, 3, 4, 5]
    result1 = sum_of_list(test_list1)
    print(f"The sum of the list {test_list1} is {result1}")
    
    # Test Case 2
    test_list2 = [10, -5, 7, 3, 20]
    result2 = sum_of_list(test_list2)
    print(f"The sum of the list {test_list2} is {result2}")
    
    

## 3 实现一个多动作Agent

- 背景介绍：使用MetaGPT框架，实现一个多动作Agent，根据用户需求生成代码。
- 需求分析：通过自然语言编写代码，并且生成的代码立即执行。

### 3.1 编写SimpleRunCode动作


```python
import subprocess

class SimpleRunCode(Action):

    name: str = "SimpleRunCode"

    async def run(self, code_text: str):
        import sys
        result = subprocess.run([sys.executable, "-c", code_text], capture_output=True, text=True)
        code_result = result.stdout
        logger.info(f"{code_result=}")
        return code_result
```

### 3.2 定义RunableCoder角色


```python
class RunnableCoder(Role):

    name: str = "Alice"
    profile: str = "RunnableCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化Action
        self._init_actions([SimpleWriteCode, SimpleRunCode])
        # 设置by_order模式，顺序执行Action
        self._set_react_mode(react_mode="by_order")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        # 通过在底层按顺序选择动作
        # todo 首先是 SimpleWriteCode() 然后是 SimpleRunCode()
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0] # 得到最相似的 k 条消息
        result = await todo.run(msg.content)

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg
```

### 3.3 运行RunnableCoder角色


```python
async def main():
    msg = "write a function that calculates the sum of a list"
    role = RunnableCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)
```


```python
await main()
```

    2024-02-29 16:11:00.965 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-02-29 16:11:00.966 | INFO     | __main__:main:4 - write a function that calculates the sum of a list
    2024-02-29 16:11:00.967 | INFO     | __main__:_act:14 - Alice(RunnableCoder): ready to SimpleWriteCode
    

     ```python
    def sum_of_list(lst):
        return sum(lst)
    
    # Test Case 1
    test_list1 = [1, 2, 3, 4, 5]
    result1 = sum_of_list(test_list1)
    print(f"The sum of the list {test_list1} is {result1}")
    
    # Test Case 2
    test_list2 = [10, -5, 7, 3, 20]
    result2 = sum_of_list(test_list2)
    print(f"The sum of the list {test_list2} is {result2}")
    ```

    2024-02-29 16:11:09.427 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.002 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 43, completion_tokens: 157
    2024-02-29 16:11:09.428 | INFO     | __main__:_act:14 - Alice(RunnableCoder): ready to SimpleRunCode
    2024-02-29 16:11:09.537 | INFO     | __main__:run:11 - code_result='The sum of the list [1, 2, 3, 4, 5] is 15\nThe sum of the list [10, -5, 7, 3, 20] is 35\n'
    2024-02-29 16:11:09.538 | INFO     | __main__:main:6 - RunnableCoder: The sum of the list [1, 2, 3, 4, 5] is 15
    The sum of the list [10, -5, 7, 3, 20] is 35
    
    

## 4 实现更复杂的Agent：技术文件助手

- 背景介绍：使用MetaGPT实现一个更复杂的Agent，将需求进行拆解。
- 需求分析：实现一个技术文件助手，获取用户输入的需求，生成文档大纲，将任务根据大纲拆解出子任务，然后逐步完成各个子任务，最后将任务汇总输出完整文档。

### 4.1 编写WriteDirectory动作


```python
from datetime import datetime
from typing import Dict

from metagpt.const import TUTORIAL_PATH
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.utils.file import File

from metagpt.actions import Action
from metagpt.utils.common import OutputParser
```


```python
COMMON_PROMPT = """
You are now a seasoned technical professional in the field of the internet. 
We need you to write a technical tutorial with the topic "{topic}".
"""

DIRECTORY_PROMPT = (
    COMMON_PROMPT
    + """
Please provide the specific table of contents for this tutorial, strictly following the following requirements:
1. The output must be strictly in the specified language, {language}.
2. Answer strictly in the dictionary format like {{"title": "xxx", "directory": [{{"dir 1": ["sub dir 1", "sub dir 2"]}}, {{"dir 2": ["sub dir 3", "sub dir 4"]}}]}}.
3. The Answer strictl use English punctuation.
4. The directory should be as specific and sufficient as possible, with a primary and secondary directory.The secondary directory is in the array.
5. Do not have extra spaces or line breaks.
6. Each directory title has practical significance.
"""
)

CONTENT_PROMPT = (
    COMMON_PROMPT
    + """
Now I will give you the module directory titles for the topic. 
Please output the detailed principle content of this title in detail. 
If there are code examples, please provide them according to standard code specifications. 
Without a code example, it is not necessary.

The module directory titles for the topic is as follows:
{directory}

Strictly limit output according to the following requirements:
1. Follow the Markdown syntax format for layout.
2. If there are code examples, they must follow standard syntax specifications, have document annotations, and be displayed in code blocks.
3. The output must be strictly in the specified language, {language}.
4. Do not have redundant output, including concluding remarks.
5. Strict requirement not to output the topic "{topic}".
"""
)
```


```python
class WriteDirectory(Action):
    """Action class for writing tutorial directories.

    Args:
        name: The name of the action.
        language: The language to output, default is "Chinese".
        
        用于编写教程目录的动作类。
        参数：
        name：动作的名称。
        language：输出的语言，默认为"Chinese"。
    """

    name: str = "WriteDirectory"
    language: str = "Chinese"

    async def run(self, topic: str, *args, **kwargs) -> Dict:
        """Execute the action to generate a tutorial directory according to the topic.

        Args:
            topic: The tutorial topic.

        Returns:
            the tutorial directory information, including {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}.
        """
        prompt = DIRECTORY_PROMPT.format(topic=topic, language=self.language)
        resp = await self._aask(prompt=prompt)
        return OutputParser.extract_struct(resp, dict)
```

### 4.2 编写WriteContent动作


```python
class WriteContent(Action):
    """Action class for writing tutorial content.

    Args:
        name: The name of the action.
        directory: The content to write.
        language: The language to output, default is "Chinese".
    """

    name: str = "WriteContent"
    directory: dict = dict()
    language: str = "Chinese"

    async def run(self, topic: str, *args, **kwargs) -> str:
        """Execute the action to write document content according to the directory and topic.

        Args:
            topic: The tutorial topic.

        Returns:
            The written tutorial content.
        """
        prompt = CONTENT_PROMPT.format(
            topic=topic, language=self.language, directory=self.directory)
        return await self._aask(prompt=prompt)
```

### 4.3 编写TutorialAssistant角色


```python
class TutorialAssistant(Role):
    """Tutorial assistant, input one sentence to generate a tutorial document in markup format.

    Args:
        name: The name of the role.
        profile: The role profile description.
        goal: The goal of the role.
        constraints: Constraints or requirements for the role.
        language: The language in which the tutorial documents will be generated.
    """

    name: str = "Stitch"
    profile: str = "Tutorial Assistant"
    goal: str = "Generate tutorial documents"
    # 内容以markdown格式输出
    constraints: str = "Strictly follow Markdown's syntax, with neat and standardized layout"
    language: str = "Chinese"

    topic: str = ""
    main_title: str = ""
    total_content: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([WriteDirectory(language=self.language)])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        
    async def _think(self) -> None:
        """Determine the next action to be taken by the role."""
        logger.info(self.rc.state)
        logger.info(self,)
        # 如果没有动作需要执行，则执行第一个动作
        if self.rc.todo is None:
            self._set_state(0)
            return
        
        # 继续执行下一个动作
        if self.rc.state + 1 < len(self.states):
            self._set_state(self.rc.state + 1)
        else:
            self.rc.todo = None

    async def _handle_directory(self, titles: Dict) -> Message:
        """Handle the directories for the tutorial document.

        Args:
            titles: A dictionary containing the titles and directory structure,
                    such as {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}

        Returns:
            A message containing information about the directory.
        """
        self.main_title = titles.get("title")
        directory = f"{self.main_title}\n"
        self.total_content += f"# {self.main_title}"
        actions = list()
        for first_dir in titles.get("directory"):
            # 添加生成子任务内容的动作
            actions.append(WriteContent(
                language=self.language, directory=first_dir))
            key = list(first_dir.keys())[0]
            directory += f"- {key}\n"
            for second_dir in first_dir[key]:
                directory += f"  - {second_dir}\n"
        # 更新action列表        
        self._init_actions(actions)
        self.rc.todo = None
        return Message(content=directory)
    
    async def _act(self) -> Message:
        """Perform an action as determined by the role.

        Returns:
            A message containing the result of the action.
        """
        todo = self.rc.todo        
        if type(todo) is WriteDirectory:
            # 生成目录
            msg = self.rc.memory.get(k=1)[0]
            self.topic = msg.content
            resp = await todo.run(topic=self.topic)
            logger.info(resp)
            # 根据目录内容，生成子任务
            return await self._handle_directory(resp)
        resp = await todo.run(topic=self.topic)
        logger.info(resp)
        if self.total_content != "":
            self.total_content += "\n\n\n"
        self.total_content += resp
        return Message(content=resp, role=self.profile)
    
    async def react(self) -> Message:
        """Execute the assistant's think and actions.

        Returns:
            A message containing the final result of the assistant's actions.
        """
        while True:
            # 先思考
            await self._think()
            if self.rc.todo is None:
                break
            # 再执行动作    
            msg = await self._act()
        
        root_path = TUTORIAL_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        await File.write(root_path, f"{self.main_title}.md", self.total_content.encode('utf-8'))
        return msg
```

### 4.5 运行TutorialAssistant角色


```python
async def main():
    msg = "Git 教程"
    role = TutorialAssistant()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)
```


```python
await main()
```

    2024-02-29 18:02:35.973 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-02-29 18:02:35.974 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-02-29 18:02:35.975 | INFO     | __main__:main:4 - Git 教程
    2024-02-29 18:02:35.975 | INFO     | __main__:_think:30 - -1
    2024-02-29 18:02:35.976 | INFO     | __main__:_think:31 - name='Stitch' profile='Tutorial Assistant' goal='Generate tutorial documents' constraints="Strictly follow Markdown's syntax, with neat and standardized layout" desc='' is_human=False llm=<metagpt.provider.zhipuai_api.ZhiPuAILLM object at 0x0000026CDF9BB100> role_id='' states=['0. WriteDirectory'] actions=[WriteDirectory] rc=RoleContext(env=None, msg_buffer=MessageQueue(), memory=Memory(storage=[user: Git 教程], index=defaultdict(<class 'list'>, {'metagpt.actions.add_requirement.UserRequirement': [user: Git 教程]}), ignore_id=False), state=-1, todo=None, watch={'metagpt.actions.add_requirement.UserRequirement'}, news=[user: Git 教程], react_mode='by_order', max_react_loop=1) subscription={'__main__.TutorialAssistant', 'Stitch'} recovered=False latest_observed_msg=user: Git 教程 language='Chinese' topic='' main_title='' total_content=''
    

     {
      "title": "Git 教程",
      "directory": [
        {
          "dir 1": [
            "sub dir 1: Git 简介",
            "sub dir 2: Git 安装与配置"
          ]
        },
        {
          "dir 2": [
            "sub dir 3: Git 基本操作",
            "sub dir 4: Git 分支管理",
            "sub dir 5: Git 代码提交与拉取",
            "sub dir 6: Git 冲突解决",
            "sub dir 7: Git 标签管理",
            "sub dir 8: Git 高级功能与应用"
          ]
        }
      ]

    2024-02-29 18:02:45.086 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 191, completion_tokens: 165
    2024-02-29 18:02:45.088 | INFO     | __main__:_act:82 - {'title': 'Git 教程', 'directory': [{'dir 1': ['sub dir 1: Git 简介', 'sub dir 2: Git 安装与配置']}, {'dir 2': ['sub dir 3: Git 基本操作', 'sub dir 4: Git 分支管理', 'sub dir 5: Git 代码提交与拉取', 'sub dir 6: Git 冲突解决', 'sub dir 7: Git 标签管理', 'sub dir 8: Git 高级功能与应用']}]}
    2024-02-29 18:02:45.089 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-02-29 18:02:45.089 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-02-29 18:02:45.090 | INFO     | __main__:_think:30 - 0
    2024-02-29 18:02:45.091 | INFO     | __main__:_think:31 - name='Stitch' profile='Tutorial Assistant' goal='Generate tutorial documents' constraints="Strictly follow Markdown's syntax, with neat and standardized layout" desc='' is_human=False llm=<metagpt.provider.zhipuai_api.ZhiPuAILLM object at 0x0000026CDF9BB100> role_id='' states=['0. WriteContent', '1. WriteContent'] actions=[WriteContent, WriteContent] rc=RoleContext(env=None, msg_buffer=MessageQueue(), memory=Memory(storage=[user: Git 教程], index=defaultdict(<class 'list'>, {'metagpt.actions.add_requirement.UserRequirement': [user: Git 教程]}), ignore_id=False), state=0, todo=None, watch={'metagpt.actions.add_requirement.UserRequirement'}, news=[user: Git 教程], react_mode='by_order', max_react_loop=1) subscription={'__main__.TutorialAssistant', 'Stitch'} recovered=False latest_observed_msg=user: Git 教程 language='Chinese' topic='Git 教程' main_title='Git 教程' total_content='# Git 教程'
    

    
    } ```
    # Git 简介
    
    Git 是一款开源的分布式版本控制系统，用于追踪文件更改和协调多人之间的工作。它最初是由 Linus Torvalds 开发的，用于管理 Linux 内核的源代码。Git 具有高效、安全、可扩展等特点，已成为当今最受欢迎的版本控制系统。
    
    Git 的主要特点有：
    
    1. 快：Git 具有良好的性能，特别是在处理大型项目时表现出色。
    
    2. 安全：Git 提供了数据完整性，通过哈希树（Merkle Tree）结构确保数据不易损坏。
    
    3. 可扩展：Git 支持插件和扩展，可以根据需求定制功能。
    
    4. 分支和合并：Git 支持分支操作，方便开发者在不同功能分支上进行开发，然后将分支合并到主分支。
    
    5. 压缩存储：Git 采用数据压缩算法，减少存储空间占用。
    
    6. 远程协作：Git 支持远程仓库，方便团队成员之间协作开发。
    
    ## Git 安装与配置
    
    ### 安装 Git
    
    在我国，推荐使用 Git 社区版（Git-cmd）进行安装。请按照以下步骤进行安装：
    
    1. 打开终端。
    
    2. 输入以下命令下载 Git：
    
    ```
    wget -c https://github.com/git-scm/git/releases/download/v2.34.1/git-2.34.1-64-bit-posix-seh-rt_64.tar.gz
    ```
    
    3. 解压下载的文件：
    
    ```
    tar -zxvf git-2.34.1-64-bit-posix-seh-rt_64.tar.gz
    ```
    
    4. 进入解压后的目录：
    
    ```
    cd git-2.34.1
    ```
    
    5. 编译并安装 Git：
    
    ```
    ./configure
    make
    sudo make install
    ```
    
    6. 安装完成后，检查 Git 是否安装成功：
    
    ```
    git --version
    ```
    
    如果显示 Git 版本信息，说明安装成功。
    
    ### 配置 Git
    
    1. 设置 Git 用户名和邮箱：
    
    ```
    git config --global user.name "你的名字"
    git config --global user.email "你的邮箱"
    ```
    
    2. 设置 Git 参数界面：
    
    ```
    git config --global gitprompt.show '%s'
    ```
    
    3. 设置 Git 分支参数：
    
    ```
    git config --global branch.autosync true
    git config --global branch.default master
    ```
    
    4. 设置 Git 远程仓库：
    
    ```
    git remote add origin https://github.com/你的用户名/你的仓库名.git
    ```
    
    5. 创建 .gitignore 文件，忽略编译产物和日志文件：
    
    ```
    touch .gitignore
    echo "# 编译产物" >> .gitignore
    echo "# 日志文件" >> .gitignore
    ```
    
    6. 创建 .gitconfig 文件，配置 Git 参数：
    
    ```
    touch .gitconfig
    echo "[core]" >> .gitconfig
    echo "autocrlf = true" >> .gitconfig
    echo "filemode = true" >> .gitconfig
    ```
    
    至此，Git 安装与配置完成。接下来，您可以开始使用 Git 进行版本控制。

    2024-02-29 18:03:29.186 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.001, prompt_tokens: 236, completion_tokens: 804
    2024-02-29 18:03:29.187 | INFO     | __main__:_act:86 -  ```
    # Git 简介
    
    Git 是一款开源的分布式版本控制系统，用于追踪文件更改和协调多人之间的工作。它最初是由 Linus Torvalds 开发的，用于管理 Linux 内核的源代码。Git 具有高效、安全、可扩展等特点，已成为当今最受欢迎的版本控制系统。
    
    Git 的主要特点有：
    
    1. 快：Git 具有良好的性能，特别是在处理大型项目时表现出色。
    
    2. 安全：Git 提供了数据完整性，通过哈希树（Merkle Tree）结构确保数据不易损坏。
    
    3. 可扩展：Git 支持插件和扩展，可以根据需求定制功能。
    
    4. 分支和合并：Git 支持分支操作，方便开发者在不同功能分支上进行开发，然后将分支合并到主分支。
    
    5. 压缩存储：Git 采用数据压缩算法，减少存储空间占用。
    
    6. 远程协作：Git 支持远程仓库，方便团队成员之间协作开发。
    
    ## Git 安装与配置
    
    ### 安装 Git
    
    在我国，推荐使用 Git 社区版（Git-cmd）进行安装。请按照以下步骤进行安装：
    
    1. 打开终端。
    
    2. 输入以下命令下载 Git：
    
    ```
    wget -c https://github.com/git-scm/git/releases/download/v2.34.1/git-2.34.1-64-bit-posix-seh-rt_64.tar.gz
    ```
    
    3. 解压下载的文件：
    
    ```
    tar -zxvf git-2.34.1-64-bit-posix-seh-rt_64.tar.gz
    ```
    
    4. 进入解压后的目录：
    
    ```
    cd git-2.34.1
    ```
    
    5. 编译并安装 Git：
    
    ```
    ./configure
    make
    sudo make install
    ```
    
    6. 安装完成后，检查 Git 是否安装成功：
    
    ```
    git --version
    ```
    
    如果显示 Git 版本信息，说明安装成功。
    
    ### 配置 Git
    
    1. 设置 Git 用户名和邮箱：
    
    ```
    git config --global user.name "你的名字"
    git config --global user.email "你的邮箱"
    ```
    
    2. 设置 Git 参数界面：
    
    ```
    git config --global gitprompt.show '%s'
    ```
    
    3. 设置 Git 分支参数：
    
    ```
    git config --global branch.autosync true
    git config --global branch.default master
    ```
    
    4. 设置 Git 远程仓库：
    
    ```
    git remote add origin https://github.com/你的用户名/你的仓库名.git
    ```
    
    5. 创建 .gitignore 文件，忽略编译产物和日志文件：
    
    ```
    touch .gitignore
    echo "# 编译产物" >> .gitignore
    echo "# 日志文件" >> .gitignore
    ```
    
    6. 创建 .gitconfig 文件，配置 Git 参数：
    
    ```
    touch .gitconfig
    echo "[core]" >> .gitconfig
    echo "autocrlf = true" >> .gitconfig
    echo "filemode = true" >> .gitconfig
    ```
    
    至此，Git 安装与配置完成。接下来，您可以开始使用 Git 进行版本控制。
    2024-02-29 18:03:29.189 | INFO     | __main__:_think:30 - 0
    2024-02-29 18:03:29.190 | INFO     | __main__:_think:31 - name='Stitch' profile='Tutorial Assistant' goal='Generate tutorial documents' constraints="Strictly follow Markdown's syntax, with neat and standardized layout" desc='' is_human=False llm=<metagpt.provider.zhipuai_api.ZhiPuAILLM object at 0x0000026CDF9BB100> role_id='' states=['0. WriteContent', '1. WriteContent'] actions=[WriteContent, WriteContent] rc=RoleContext(env=None, msg_buffer=MessageQueue(), memory=Memory(storage=[user: Git 教程], index=defaultdict(<class 'list'>, {'metagpt.actions.add_requirement.UserRequirement': [user: Git 教程]}), ignore_id=False), state=0, todo=WriteContent, watch={'metagpt.actions.add_requirement.UserRequirement'}, news=[user: Git 教程], react_mode='by_order', max_react_loop=1) subscription={'__main__.TutorialAssistant', 'Stitch'} recovered=False latest_observed_msg=user: Git 教程 language='Chinese' topic='Git 教程' main_title='Git 教程' total_content='# Git 教程\n\n\n ```\n# Git 简介\n\nGit 是一款开源的分布式版本控制系统，用于追踪文件更改和协调多人之间的工作。它最初是由 Linus Torvalds 开发的，用于管理 Linux 内核的源代码。Git 具有高效、安全、可扩展等特点，已成为当今最受欢迎的版本控制系统。\n\nGit 的主要特点有：\n\n1. 快：Git 具有良好的性能，特别是在处理大型项目时表现出色。\n\n2. 安全：Git 提供了数据完整性，通过哈希树（Merkle Tree）结构确保数据不易损坏。\n\n3. 可扩展：Git 支持插件和扩展，可以根据需求定制功能。\n\n4. 分支和合并：Git 支持分支操作，方便开发者在不同功能分支上进行开发，然后将分支合并到主分支。\n\n5. 压缩存储：Git 采用数据压缩算法，减少存储空间占用。\n\n6. 远程协作：Git 支持远程仓库，方便团队成员之间协作开发。\n\n## Git 安装与配置\n\n### 安装 Git\n\n在我国，推荐使用 Git 社区版（Git-cmd）进行安装。请按照以下步骤进行安装：\n\n1. 打开终端。\n\n2. 输入以下命令下载 Git：\n\n```\nwget -c https://github.com/git-scm/git/releases/download/v2.34.1/git-2.34.1-64-bit-posix-seh-rt_64.tar.gz\n```\n\n3. 解压下载的文件：\n\n```\ntar -zxvf git-2.34.1-64-bit-posix-seh-rt_64.tar.gz\n```\n\n4. 进入解压后的目录：\n\n```\ncd git-2.34.1\n```\n\n5. 编译并安装 Git：\n\n```\n./configure\nmake\nsudo make install\n```\n\n6. 安装完成后，检查 Git 是否安装成功：\n\n```\ngit --version\n```\n\n如果显示 Git 版本信息，说明安装成功。\n\n### 配置 Git\n\n1. 设置 Git 用户名和邮箱：\n\n```\ngit config --global user.name "你的名字"\ngit config --global user.email "你的邮箱"\n```\n\n2. 设置 Git 参数界面：\n\n```\ngit config --global gitprompt.show \'%s\'\n```\n\n3. 设置 Git 分支参数：\n\n```\ngit config --global branch.autosync true\ngit config --global branch.default master\n```\n\n4. 设置 Git 远程仓库：\n\n```\ngit remote add origin https://github.com/你的用户名/你的仓库名.git\n```\n\n5. 创建 .gitignore 文件，忽略编译产物和日志文件：\n\n```\ntouch .gitignore\necho "# 编译产物" >> .gitignore\necho "# 日志文件" >> .gitignore\n```\n\n6. 创建 .gitconfig 文件，配置 Git 参数：\n\n```\ntouch .gitconfig\necho "[core]" >> .gitconfig\necho "autocrlf = true" >> .gitconfig\necho "filemode = true" >> .gitconfig\n```\n\n至此，Git 安装与配置完成。接下来，您可以开始使用 Git 进行版本控制。'
    

     ```
    # Git 教程
    
    ## Git 基本操作
    
    ### 1.1 初始化仓库
    
    ```bash
    git init
    ```
    
    ### 1.2 添加文件
    
    ```bash
    git add 文件名
    ```
    
    ### 1.3 提交代码
    
    ```bash
    git commit -m "提交信息"
    ```
    
    ### 1.4 创建分支
    
    ```bash
    git checkout -b 分支名
    ```
    
    ### 1.5 切换分支
    
    ```bash
    git checkout 分支名
    ```
    
    ### 1.6 查看分支
    
    ```bash
    git branch
    ```
    
    ### 1.7 合并分支
    
    ```bash
    git merge 分支名
    ```
    
    ### 1.8 删除分支
    
    ```bash
    git branch -d 分支名
    ```
    
    ## Git 分支管理
    
    ### 2.1 分支策略
    
    - 主分支：master/main
    - 开发分支：develop
    - 功能分支：feature
    - 发布分支：release
    -  hotfix 分支：hotfix
    
    ### 2.2 分支命名规范
    
    - 以字母开头，可以包含数字和字母
    - 推荐使用短命名，便于识别
    - 避免使用中文和特殊字符
    
    ### 2.3 分支管理流程
    
    1. 从主分支创建开发分支
    2. 开发过程中，不断提交代码，合并到开发分支
    3. 开发分支达到一定稳定性，创建发布分支
    4. 发布分支进行测试，修复 bug，合并到发布分支
    5. 发布分支稳定后，合并到主分支
    6. 主分支合并完成后，删除开发分支和发布分支
    
    ## Git 代码提交与拉取
    
    ### 3.1 代码提交
    
    - 提交时添加描述性的提交信息，便于后人查看
    - 遵循 [Conventional Changelog](https://conventionalchangelog.com/) 规范
    
    ### 3.2 代码拉取
    
    ```bash
    git pull
    ```
    
    ## Git 冲突解决
    
    ### 4.1 冲突检测
    
    - 相同文件名，内容不同：冲突
    - 相同文件名，内容相同：未冲突
    - 不同文件名：未冲突
    
    ### 4.2 冲突解决
    
    1. 手动解决：比较冲突文件，修改后合并
    2. 使用 [`git-median`](https://github.com/ginatls/git-median) 工具自动解决冲突
    
    ## Git 标签管理
    
    ### 5.1 创建标签
    
    ```bash
    git tag -a 标签名 -m "描述"
    ```
    
    ### 5.2 推送标签
    
    ```bash
    git push origin 标签名
    ```
    
    ### 5.3 拉取标签
    
    ```bash
    git pull origin 标签名
    ```
    
    ### 5.4 删除标签
    
    ```bash
    git tag -d 标签名
    ```
    
    ## Git 高级功能与应用
    
    ### 6.1 远程仓库管理
    
    1. 添加远程仓库
    
    ```bash
    git remote add 仓库名 远程仓库地址
    ```
    
    2. 删除远程仓库
    
    ```bash
    git remote remove 仓库名
    ```
    
    3. 推送/拉取仓库
    
    ```bash
    git push
    git pull
    ```
    
    ### 6.2 代码审查
    
    1. 创建审查分支
    
    ```bash
    git checkout -b 审查分支名
    ```
    
    2. 提交代码审查请求
    
    ```bash
    git review
    ```
    
    3. 审查代码
    
    - 查看代码：`git log 审查分支名`
    - 提交反馈：`git review <反馈编号>`
    
    ### 6.3 代码部署
    
    1. 创建部署脚本
    
    ```bash
    git deploy
    ```
    
    2. 部署代码
    
    ```bash
    ./部署脚本
    ```
    
    ### 6.4 代码备份
    
    1. 创建备份脚本
    
    ```bash
    git backup
    ```
    
    2. 备份代码
    
    ```bash

    2024-02-29 18:04:24.255 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.001, prompt_tokens: 284, completion_tokens: 1024
    2024-02-29 18:04:24.256 | INFO     | __main__:_act:86 -  ```
    # Git 教程
    
    ## Git 基本操作
    
    ### 1.1 初始化仓库
    
    ```bash
    git init
    ```
    
    ### 1.2 添加文件
    
    ```bash
    git add 文件名
    ```
    
    ### 1.3 提交代码
    
    ```bash
    git commit -m "提交信息"
    ```
    
    ### 1.4 创建分支
    
    ```bash
    git checkout -b 分支名
    ```
    
    ### 1.5 切换分支
    
    ```bash
    git checkout 分支名
    ```
    
    ### 1.6 查看分支
    
    ```bash
    git branch
    ```
    
    ### 1.7 合并分支
    
    ```bash
    git merge 分支名
    ```
    
    ### 1.8 删除分支
    
    ```bash
    git branch -d 分支名
    ```
    
    ## Git 分支管理
    
    ### 2.1 分支策略
    
    - 主分支：master/main
    - 开发分支：develop
    - 功能分支：feature
    - 发布分支：release
    -  hotfix 分支：hotfix
    
    ### 2.2 分支命名规范
    
    - 以字母开头，可以包含数字和字母
    - 推荐使用短命名，便于识别
    - 避免使用中文和特殊字符
    
    ### 2.3 分支管理流程
    
    1. 从主分支创建开发分支
    2. 开发过程中，不断提交代码，合并到开发分支
    3. 开发分支达到一定稳定性，创建发布分支
    4. 发布分支进行测试，修复 bug，合并到发布分支
    5. 发布分支稳定后，合并到主分支
    6. 主分支合并完成后，删除开发分支和发布分支
    
    ## Git 代码提交与拉取
    
    ### 3.1 代码提交
    
    - 提交时添加描述性的提交信息，便于后人查看
    - 遵循 [Conventional Changelog](https://conventionalchangelog.com/) 规范
    
    ### 3.2 代码拉取
    
    ```bash
    git pull
    ```
    
    ## Git 冲突解决
    
    ### 4.1 冲突检测
    
    - 相同文件名，内容不同：冲突
    - 相同文件名，内容相同：未冲突
    - 不同文件名：未冲突
    
    ### 4.2 冲突解决
    
    1. 手动解决：比较冲突文件，修改后合并
    2. 使用 [`git-median`](https://github.com/ginatls/git-median) 工具自动解决冲突
    
    ## Git 标签管理
    
    ### 5.1 创建标签
    
    ```bash
    git tag -a 标签名 -m "描述"
    ```
    
    ### 5.2 推送标签
    
    ```bash
    git push origin 标签名
    ```
    
    ### 5.3 拉取标签
    
    ```bash
    git pull origin 标签名
    ```
    
    ### 5.4 删除标签
    
    ```bash
    git tag -d 标签名
    ```
    
    ## Git 高级功能与应用
    
    ### 6.1 远程仓库管理
    
    1. 添加远程仓库
    
    ```bash
    git remote add 仓库名 远程仓库地址
    ```
    
    2. 删除远程仓库
    
    ```bash
    git remote remove 仓库名
    ```
    
    3. 推送/拉取仓库
    
    ```bash
    git push
    git pull
    ```
    
    ### 6.2 代码审查
    
    1. 创建审查分支
    
    ```bash
    git checkout -b 审查分支名
    ```
    
    2. 提交代码审查请求
    
    ```bash
    git review
    ```
    
    3. 审查代码
    
    - 查看代码：`git log 审查分支名`
    - 提交反馈：`git review <反馈编号>`
    
    ### 6.3 代码部署
    
    1. 创建部署脚本
    
    ```bash
    git deploy
    ```
    
    2. 部署代码
    
    ```bash
    ./部署脚本
    ```
    
    ### 6.4 代码备份
    
    1. 创建备份脚本
    
    ```bash
    git backup
    ```
    
    2. 备份代码
    
    ```bash
    2024-02-29 18:04:24.256 | INFO     | __main__:_think:30 - 1
    2024-02-29 18:04:24.257 | INFO     | __main__:_think:31 - name='Stitch' profile='Tutorial Assistant' goal='Generate tutorial documents' constraints="Strictly follow Markdown's syntax, with neat and standardized layout" desc='' is_human=False llm=<metagpt.provider.zhipuai_api.ZhiPuAILLM object at 0x0000026CDF9BB100> role_id='' states=['0. WriteContent', '1. WriteContent'] actions=[WriteContent, WriteContent] rc=RoleContext(env=None, msg_buffer=MessageQueue(), memory=Memory(storage=[user: Git 教程], index=defaultdict(<class 'list'>, {'metagpt.actions.add_requirement.UserRequirement': [user: Git 教程]}), ignore_id=False), state=1, todo=WriteContent, watch={'metagpt.actions.add_requirement.UserRequirement'}, news=[user: Git 教程], react_mode='by_order', max_react_loop=1) subscription={'__main__.TutorialAssistant', 'Stitch'} recovered=False latest_observed_msg=user: Git 教程 language='Chinese' topic='Git 教程' main_title='Git 教程' total_content='# Git 教程\n\n\n ```\n# Git 简介\n\nGit 是一款开源的分布式版本控制系统，用于追踪文件更改和协调多人之间的工作。它最初是由 Linus Torvalds 开发的，用于管理 Linux 内核的源代码。Git 具有高效、安全、可扩展等特点，已成为当今最受欢迎的版本控制系统。\n\nGit 的主要特点有：\n\n1. 快：Git 具有良好的性能，特别是在处理大型项目时表现出色。\n\n2. 安全：Git 提供了数据完整性，通过哈希树（Merkle Tree）结构确保数据不易损坏。\n\n3. 可扩展：Git 支持插件和扩展，可以根据需求定制功能。\n\n4. 分支和合并：Git 支持分支操作，方便开发者在不同功能分支上进行开发，然后将分支合并到主分支。\n\n5. 压缩存储：Git 采用数据压缩算法，减少存储空间占用。\n\n6. 远程协作：Git 支持远程仓库，方便团队成员之间协作开发。\n\n## Git 安装与配置\n\n### 安装 Git\n\n在我国，推荐使用 Git 社区版（Git-cmd）进行安装。请按照以下步骤进行安装：\n\n1. 打开终端。\n\n2. 输入以下命令下载 Git：\n\n```\nwget -c https://github.com/git-scm/git/releases/download/v2.34.1/git-2.34.1-64-bit-posix-seh-rt_64.tar.gz\n```\n\n3. 解压下载的文件：\n\n```\ntar -zxvf git-2.34.1-64-bit-posix-seh-rt_64.tar.gz\n```\n\n4. 进入解压后的目录：\n\n```\ncd git-2.34.1\n```\n\n5. 编译并安装 Git：\n\n```\n./configure\nmake\nsudo make install\n```\n\n6. 安装完成后，检查 Git 是否安装成功：\n\n```\ngit --version\n```\n\n如果显示 Git 版本信息，说明安装成功。\n\n### 配置 Git\n\n1. 设置 Git 用户名和邮箱：\n\n```\ngit config --global user.name "你的名字"\ngit config --global user.email "你的邮箱"\n```\n\n2. 设置 Git 参数界面：\n\n```\ngit config --global gitprompt.show \'%s\'\n```\n\n3. 设置 Git 分支参数：\n\n```\ngit config --global branch.autosync true\ngit config --global branch.default master\n```\n\n4. 设置 Git 远程仓库：\n\n```\ngit remote add origin https://github.com/你的用户名/你的仓库名.git\n```\n\n5. 创建 .gitignore 文件，忽略编译产物和日志文件：\n\n```\ntouch .gitignore\necho "# 编译产物" >> .gitignore\necho "# 日志文件" >> .gitignore\n```\n\n6. 创建 .gitconfig 文件，配置 Git 参数：\n\n```\ntouch .gitconfig\necho "[core]" >> .gitconfig\necho "autocrlf = true" >> .gitconfig\necho "filemode = true" >> .gitconfig\n```\n\n至此，Git 安装与配置完成。接下来，您可以开始使用 Git 进行版本控制。\n\n\n ```\n# Git 教程\n\n## Git 基本操作\n\n### 1.1 初始化仓库\n\n```bash\ngit init\n```\n\n### 1.2 添加文件\n\n```bash\ngit add 文件名\n```\n\n### 1.3 提交代码\n\n```bash\ngit commit -m "提交信息"\n```\n\n### 1.4 创建分支\n\n```bash\ngit checkout -b 分支名\n```\n\n### 1.5 切换分支\n\n```bash\ngit checkout 分支名\n```\n\n### 1.6 查看分支\n\n```bash\ngit branch\n```\n\n### 1.7 合并分支\n\n```bash\ngit merge 分支名\n```\n\n### 1.8 删除分支\n\n```bash\ngit branch -d 分支名\n```\n\n## Git 分支管理\n\n### 2.1 分支策略\n\n- 主分支：master/main\n- 开发分支：develop\n- 功能分支：feature\n- 发布分支：release\n-  hotfix 分支：hotfix\n\n### 2.2 分支命名规范\n\n- 以字母开头，可以包含数字和字母\n- 推荐使用短命名，便于识别\n- 避免使用中文和特殊字符\n\n### 2.3 分支管理流程\n\n1. 从主分支创建开发分支\n2. 开发过程中，不断提交代码，合并到开发分支\n3. 开发分支达到一定稳定性，创建发布分支\n4. 发布分支进行测试，修复 bug，合并到发布分支\n5. 发布分支稳定后，合并到主分支\n6. 主分支合并完成后，删除开发分支和发布分支\n\n## Git 代码提交与拉取\n\n### 3.1 代码提交\n\n- 提交时添加描述性的提交信息，便于后人查看\n- 遵循 [Conventional Changelog](https://conventionalchangelog.com/) 规范\n\n### 3.2 代码拉取\n\n```bash\ngit pull\n```\n\n## Git 冲突解决\n\n### 4.1 冲突检测\n\n- 相同文件名，内容不同：冲突\n- 相同文件名，内容相同：未冲突\n- 不同文件名：未冲突\n\n### 4.2 冲突解决\n\n1. 手动解决：比较冲突文件，修改后合并\n2. 使用 [`git-median`](https://github.com/ginatls/git-median) 工具自动解决冲突\n\n## Git 标签管理\n\n### 5.1 创建标签\n\n```bash\ngit tag -a 标签名 -m "描述"\n```\n\n### 5.2 推送标签\n\n```bash\ngit push origin 标签名\n```\n\n### 5.3 拉取标签\n\n```bash\ngit pull origin 标签名\n```\n\n### 5.4 删除标签\n\n```bash\ngit tag -d 标签名\n```\n\n## Git 高级功能与应用\n\n### 6.1 远程仓库管理\n\n1. 添加远程仓库\n\n```bash\ngit remote add 仓库名 远程仓库地址\n```\n\n2. 删除远程仓库\n\n```bash\ngit remote remove 仓库名\n```\n\n3. 推送/拉取仓库\n\n```bash\ngit push\ngit pull\n```\n\n### 6.2 代码审查\n\n1. 创建审查分支\n\n```bash\ngit checkout -b 审查分支名\n```\n\n2. 提交代码审查请求\n\n```bash\ngit review\n```\n\n3. 审查代码\n\n- 查看代码：`git log 审查分支名`\n- 提交反馈：`git review <反馈编号>`\n\n### 6.3 代码部署\n\n1. 创建部署脚本\n\n```bash\ngit deploy\n```\n\n2. 部署代码\n\n```bash\n./部署脚本\n```\n\n### 6.4 代码备份\n\n1. 创建备份脚本\n\n```bash\ngit backup\n```\n\n2. 备份代码\n\n```bash'
    2024-02-29 18:04:24.266 | INFO     | __main__:main:6 - Tutorial Assistant:  ```
    # Git 教程
    
    ## Git 基本操作
    
    ### 1.1 初始化仓库
    
    ```bash
    git init
    ```
    
    ### 1.2 添加文件
    
    ```bash
    git add 文件名
    ```
    
    ### 1.3 提交代码
    
    ```bash
    git commit -m "提交信息"
    ```
    
    ### 1.4 创建分支
    
    ```bash
    git checkout -b 分支名
    ```
    
    ### 1.5 切换分支
    
    ```bash
    git checkout 分支名
    ```
    
    ### 1.6 查看分支
    
    ```bash
    git branch
    ```
    
    ### 1.7 合并分支
    
    ```bash
    git merge 分支名
    ```
    
    ### 1.8 删除分支
    
    ```bash
    git branch -d 分支名
    ```
    
    ## Git 分支管理
    
    ### 2.1 分支策略
    
    - 主分支：master/main
    - 开发分支：develop
    - 功能分支：feature
    - 发布分支：release
    -  hotfix 分支：hotfix
    
    ### 2.2 分支命名规范
    
    - 以字母开头，可以包含数字和字母
    - 推荐使用短命名，便于识别
    - 避免使用中文和特殊字符
    
    ### 2.3 分支管理流程
    
    1. 从主分支创建开发分支
    2. 开发过程中，不断提交代码，合并到开发分支
    3. 开发分支达到一定稳定性，创建发布分支
    4. 发布分支进行测试，修复 bug，合并到发布分支
    5. 发布分支稳定后，合并到主分支
    6. 主分支合并完成后，删除开发分支和发布分支
    
    ## Git 代码提交与拉取
    
    ### 3.1 代码提交
    
    - 提交时添加描述性的提交信息，便于后人查看
    - 遵循 [Conventional Changelog](https://conventionalchangelog.com/) 规范
    
    ### 3.2 代码拉取
    
    ```bash
    git pull
    ```
    
    ## Git 冲突解决
    
    ### 4.1 冲突检测
    
    - 相同文件名，内容不同：冲突
    - 相同文件名，内容相同：未冲突
    - 不同文件名：未冲突
    
    ### 4.2 冲突解决
    
    1. 手动解决：比较冲突文件，修改后合并
    2. 使用 [`git-median`](https://github.com/ginatls/git-median) 工具自动解决冲突
    
    ## Git 标签管理
    
    ### 5.1 创建标签
    
    ```bash
    git tag -a 标签名 -m "描述"
    ```
    
    ### 5.2 推送标签
    
    ```bash
    git push origin 标签名
    ```
    
    ### 5.3 拉取标签
    
    ```bash
    git pull origin 标签名
    ```
    
    ### 5.4 删除标签
    
    ```bash
    git tag -d 标签名
    ```
    
    ## Git 高级功能与应用
    
    ### 6.1 远程仓库管理
    
    1. 添加远程仓库
    
    ```bash
    git remote add 仓库名 远程仓库地址
    ```
    
    2. 删除远程仓库
    
    ```bash
    git remote remove 仓库名
    ```
    
    3. 推送/拉取仓库
    
    ```bash
    git push
    git pull
    ```
    
    ### 6.2 代码审查
    
    1. 创建审查分支
    
    ```bash
    git checkout -b 审查分支名
    ```
    
    2. 提交代码审查请求
    
    ```bash
    git review
    ```
    
    3. 审查代码
    
    - 查看代码：`git log 审查分支名`
    - 提交反馈：`git review <反馈编号>`
    
    ### 6.3 代码部署
    
    1. 创建部署脚本
    
    ```bash
    git deploy
    ```
    
    2. 部署代码
    
    ```bash
    ./部署脚本
    ```
    
    ### 6.4 代码备份
    
    1. 创建备份脚本
    
    ```bash
    git backup
    ```
    
    2. 备份代码
    
    ```bash
    

生成Git教程的markdown格式的文件。

![文档生成](images/task03-01.png)
