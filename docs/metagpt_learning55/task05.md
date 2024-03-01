# Task05 多智能体开发

## 1 多智能体组件

### 1.1 Environment介绍

- Environment：主要管理Agent的活动与信息交流。
    - `desc`：环境描述
    - `roles`：当前环境中的角色
    - `members`：当前环境中的角色和对应的状态
    - `history`：记录环境中发生的消息记录

- Environment的执行过程：
    - 遍历所有的角色，按顺序执行`role`的方法。
    - `role`根据运行时的消息，存入到`role_context`的`msg_buffer`中
    - `role`的每次行动时，都先调用`_observe`接收消息，从缓冲区和其他源准备新消息进行处理，当未接收到指令时，`role`继续等待。
    - `role`执行行动。
    - 将`role`的执行结果发布到环境中。
    - 遍历所有的角色，检查它们是否订阅了这条消息，如果订阅了，则将消息存入到`role`的`msg_buffer`中。

### 1.2 实现一个简单的多智能体系统

- 项目背景：根据给定的主题，撰写一篇优美的英文诗。
- 需求分析：接收用户的需求，学生关注到布置的题目后就会开始创作，当老师发现学生写作完成后就会给学生提出意见，根据老师给出的意见，学生将修改自己的作品，直到设定循环结束。


```python
import asyncio

from metagpt.actions import Action, UserRequirement
from metagpt.const import MESSAGE_ROUTE_TO_ALL
from metagpt.environment import Environment
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
```


```python
classroom = Environment()
```


```python
# 编写诗句，并根据老师的建议修改诗句
class WritePoem(Action):
    name: str = "WritePoem"

    PROMPT_TEMPLATE: str = """
    Here is the historical conversation record : {msg} .
    Write a poem about the subject provided by human, Return only the content of the generated poem with NO other texts.
    If the teacher provides suggestions about the poem, revise the student's poem based on the suggestions and return.
    your poem:
    """

    async def run(self, msg: str):
        prompt = self.PROMPT_TEMPLATE.format(msg=msg)

        rsp = await self._aask(prompt)

        return rsp
```


```python
# 读取诗歌作品，并给出修改意见
class ReviewPoem(Action):
    name: str = "ReviewPoem"

    PROMPT_TEMPLATE: str = """

    Here is the historical conversation record : {msg} .
    Check student-created poems about the subject provided by human and give your suggestions for revisions. You prefer poems with elegant sentences and retro style.
    Return only your comments with NO other texts.
    your comments:
    """

    async def run(self, msg: str):
        prompt = self.PROMPT_TEMPLATE.format(msg=msg)

        rsp = await self._aask(prompt)

        return rsp
```


```python
# 定义学生角色
class Student(Role):
    name: str = "xiaohong"
    profile: str = "Student"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([WritePoem])
        self._watch([UserRequirement, ReviewPoem])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        todo = self.rc.todo

        msg = self.get_memories()  # 获取所有记忆
        poem_text = await WritePoem().run(msg)
        logger.info(f'student : {poem_text}')
        msg = Message(content=poem_text, role=self.profile,
                      cause_by=type(todo))

        return msg
```


```python
# 定义老师角色
class Teacher(Role):
    name: str = "wanglaoshi"
    profile: str = "Teacher"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([ReviewPoem])
        self._watch([WritePoem])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        todo = self.rc.todo

        msg = self.get_memories()  # 获取所有记忆
        poem_text = await ReviewPoem().run(msg)
        logger.info(f'teacher : {poem_text}')
        msg = Message(content=poem_text, role=self.profile,
                      cause_by=type(todo))

        return msg
```


```python
async def main(topic: str, n_round=3):
    classroom.add_roles([Student(), Teacher()])

    classroom.publish_message(
        Message(role="Human", content=topic, cause_by=UserRequirement,
                send_to='' or MESSAGE_ROUTE_TO_ALL),
        peekable=False,
    )

    while n_round > 0:
        n_round -= 1
        logger.debug(f"max {n_round=} left.")

        await classroom.run()
```


```python
await main(topic='write a poem about moon')
```

    2024-03-01 20:07:34.948 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 20:07:34.949 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 20:07:34.951 | INFO     | __main__:_act:12 - xiaohong(Student): ready to WritePoem
    2024-03-01 20:07:34.952 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    

     When the sun departs, and darkness falls,
    A celestial gem adorns thealls.
    The moon, with its radiant glow,
    Dances upon the night's velvet show.
    
    A silver disk, forever bright,
    It guides our steps along the light.
    The lunar phases, like a symphony,
    Ebb and flow, in harmony.
    
    In the stillness of the night,
    The moon's embrace enchants and delights.
    Its ethereal beams, a mistress's touch,
    Bestow upon us dreams and mystic truce.
    
    A beacon in the darkest hours,
    The moon's eternal light illuminates our souls.
    As we gaze upon its radiant face,
    We find within ourselves a guiding grace.
    
    Oh, moon of mine, eternal companion,
    Guide me through the seasons' dance and passion.
    With your silent song, inspire my heart,
    To love, to dream, and seek the truth within.
    
    In the realm of night, you reign supreme,
    A symbol of beauty, love, and harmony.
    The moon, a constant in our lives,
    Eternal, just like the mysteries of time

    2024-03-01 20:07:49.435 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 78, completion_tokens: 264
    2024-03-01 20:07:49.435 | INFO     | __main__:_act:17 - student :  When the sun departs, and darkness falls,
    A celestial gem adorns thealls.
    The moon, with its radiant glow,
    Dances upon the night's velvet show.
    
    A silver disk, forever bright,
    It guides our steps along the light.
    The lunar phases, like a symphony,
    Ebb and flow, in harmony.
    
    In the stillness of the night,
    The moon's embrace enchants and delights.
    Its ethereal beams, a mistress's touch,
    Bestow upon us dreams and mystic truce.
    
    A beacon in the darkest hours,
    The moon's eternal light illuminates our souls.
    As we gaze upon its radiant face,
    We find within ourselves a guiding grace.
    
    Oh, moon of mine, eternal companion,
    Guide me through the seasons' dance and passion.
    With your silent song, inspire my heart,
    To love, to dream, and seek the truth within.
    
    In the realm of night, you reign supreme,
    A symbol of beauty, love, and harmony.
    The moon, a constant in our lives,
    Eternal, just like the mysteries of time.
    2024-03-01 20:07:49.438 | INFO     | __main__:_act:12 - wanglaoshi(Teacher): ready to ReviewPoem
    2024-03-01 20:07:49.438 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    

    . 1. Excellent use of metaphors and personification to describe the moon's beauty and significance.
    2. The poem's structure and rhyme scheme are consistent and pleasing to the ear.
    3. The lunar phases mentioned add depth and astronomical accuracy to the poem.
    4. The poem captures the moon's enchanting presence and its role as a guide and source of inspiration.
    5. The retro style is evident in the use of descriptive words and phrases, such as "celestial gem," "silver disk," and "ethereal beams."
    6. Suggestions for revisions:
       a. Consider adding more intricate language to emphasize the retro style, such as "lunar orb" instead of "moon" and "nocturnal navel" instead of "night's velvet show."
       b. To create a more profound connection between the moon and the human soul, you may want to emphasize the moon's mystical qualities, such as its ability to inspire introspection and self-discovery.
       c. To add a touch of elegance, you could replace "bestow upon us" with "impart" or "confer."
       d. To make the final stanza more powerful, consider using a more assertive tone to emphasize the moon's reign and its role as a symbol of beauty, love, and harmony.
    
    Overall, the poem is beautifully written and captures the essence of the moon's mystique and allure. Minor revisions can further enhance its elegance and retro style.

    2024-03-01 20:08:07.090 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 357, completion_tokens: 331
    2024-03-01 20:08:07.091 | INFO     | __main__:_act:17 - teacher :  1. Excellent use of metaphors and personification to describe the moon's beauty and significance.
    2. The poem's structure and rhyme scheme are consistent and pleasing to the ear.
    3. The lunar phases mentioned add depth and astronomical accuracy to the poem.
    4. The poem captures the moon's enchanting presence and its role as a guide and source of inspiration.
    5. The retro style is evident in the use of descriptive words and phrases, such as "celestial gem," "silver disk," and "ethereal beams."
    6. Suggestions for revisions:
       a. Consider adding more intricate language to emphasize the retro style, such as "lunar orb" instead of "moon" and "nocturnal navel" instead of "night's velvet show."
       b. To create a more profound connection between the moon and the human soul, you may want to emphasize the moon's mystical qualities, such as its ability to inspire introspection and self-discovery.
       c. To add a touch of elegance, you could replace "bestow upon us" with "impart" or "confer."
       d. To make the final stanza more powerful, consider using a more assertive tone to emphasize the moon's reign and its role as a symbol of beauty, love, and harmony.
    
    Overall, the poem is beautifully written and captures the essence of the moon's mystique and allure. Minor revisions can further enhance its elegance and retro style.
    2024-03-01 20:08:07.093 | INFO     | __main__:_act:12 - xiaohong(Student): ready to WritePoem
    2024-03-01 20:08:07.094 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    

     When the sun departs, and darkness falls,
    A lunar orb adorns the nightly halls.
    The lunar phases, like a symphony,
    Ebb and flow, in cosmic harmony.
    
    A silvery disk, forever bright,
    It guides our steps along the spectral light.
    In the stillness of the night,
    The moon's embrace enchants and delights.
    
    Its ethereal beams, a mistress's touch,
    Impart upon us dreams and mystic truce.
    A beacon in the darkest hours,
    The moon's eternal light illumines our souls.
    
    As we gaze upon its radiant face,
    We find within ourselves a guiding grace.
    Oh, lunar orb, eternal companion,
    Guide me through the seasons' dance and passion.
    
    With your silent song, inspire my heart,
    To love, to dream, and seek the truth within.
    In the realm of night, you reign supreme,
    A symbol of beauty, love, and harmony.
    
    The moon, a constant in our lives,
    Eternal, just like the mysteries of time.

    2024-03-01 20:08:20.367 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 721, completion_tokens: 245
    2024-03-01 20:08:20.368 | INFO     | __main__:_act:17 - student :  When the sun departs, and darkness falls,
    A lunar orb adorns the nightly halls.
    The lunar phases, like a symphony,
    Ebb and flow, in cosmic harmony.
    
    A silvery disk, forever bright,
    It guides our steps along the spectral light.
    In the stillness of the night,
    The moon's embrace enchants and delights.
    
    Its ethereal beams, a mistress's touch,
    Impart upon us dreams and mystic truce.
    A beacon in the darkest hours,
    The moon's eternal light illumines our souls.
    
    As we gaze upon its radiant face,
    We find within ourselves a guiding grace.
    Oh, lunar orb, eternal companion,
    Guide me through the seasons' dance and passion.
    
    With your silent song, inspire my heart,
    To love, to dream, and seek the truth within.
    In the realm of night, you reign supreme,
    A symbol of beauty, love, and harmony.
    
    The moon, a constant in our lives,
    Eternal, just like the mysteries of time.
    

### 1.3 Team介绍

- Team：基于Environment之上的二次封装。
    - `investment`：用于管理团队成本（限制token消费）

- Team主要方法：
    - `hire`：用于在团队中添加员工。
    - `invest`：控制预算。

- Team的执行过程：调用`run_project`方法，给智能体们一个需求，在`n_round`循环中，重复检查预算和运行的`env`，最后返回环境中角色的历史对话。

### 1.4 基于Team实现一个智能体团队

**实现步骤：**
1. 定义每个角色能够执行的预期动作。
2. 基于标准作业程序（SOP），确保每个角色遵守。每个角色需要观察上游输出结果，并为下游发布自己的输出结果。
3. 初始化所有角色，创建一个带有环境的智能体团队，并使它们之间能够进行交互。

**角色定义：**
- `SimpleCoder`：接收用户的需求，编写主要代码。
- `SimpleTester`：获取主代码并，为其提供测试用例。
- `SimpleReviewer`：审查测试用例，检查其覆盖范围和质量。


```python
import re

import fire
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
```


```python
def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text
```


```python
class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction}.
    Return ```python``` with NO other texts,
    your code:
    """
    name: str = "SimpleWriteCode"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)

        return code_text
```


```python
class SimpleWriteTest(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Write {k} unit tests using pytest for the given function, assuming you have imported it.
    Return ```python``` with NO other texts,
    your code:
    """

    name: str = "SimpleWriteTest"

    async def run(self, context: str, k: int = 3):
        prompt = self.PROMPT_TEMPLATE.format(context=context, k=k)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)

        return code_text
```


```python
class SimpleWriteReview(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Review the test cases and provide one critical comments:
    """

    name: str = "SimpleWriteReview"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        return rsp
```


```python
class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self._init_actions([SimpleWriteCode])
```


```python
class SimpleTester(Role):
    name: str = "Bob"
    profile: str = "SimpleTester"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([SimpleWriteTest])
        self._watch([SimpleWriteCode, SimpleWriteReview])  # feel free to try this too

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        context = self.get_memories()  # use all memories as context

        code_text = await todo.run(context, k=5)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg
```


```python
class SimpleReviewer(Role):
    name: str = "Charlie"
    profile: str = "SimpleReviewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([SimpleWriteReview])
        self._watch([SimpleWriteTest])
```


```python
async def main(
        idea: str = "write a function that calculates the product of a list",
        investment: float = 3.0,
        n_round: int = 5,
        add_human: bool = False,
):
    logger.info(idea)

    team = Team()
    team.hire(
        [
            SimpleCoder(),
            SimpleTester(),
            SimpleReviewer(is_human=add_human),
        ]
    )

    team.invest(investment=investment)
    team.run_project(idea)
    await team.run(n_round=n_round)
```


```python
await main()
```

    2024-03-01 21:02:19.002 | INFO     | __main__:main:7 - write a function that calculates the product of a list
    2024-03-01 21:02:19.003 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 21:02:19.004 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 21:02:19.005 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 21:02:19.006 | INFO     | metagpt.team:invest:86 - Investment: $3.0.
    2024-03-01 21:02:19.008 | INFO     | metagpt.roles.role:_act:357 - Alice(SimpleCoder): to do SimpleWriteCode(SimpleWriteCode)
    

     def product_of_list(lst):
        return reduce(lambda x, y: x * y, lst)
    
    # Example usage:
    lst = [2, 3, 4, 5]
    result = product_of_list(lst)
    print(result)  # Output: 120

    2024-03-01 21:02:23.326 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 43, completion_tokens: 74
    2024-03-01 21:02:23.329 | INFO     | __main__:_act:11 - Bob(SimpleTester): to do SimpleWriteTest(SimpleWriteTest)
    

     ```python
    import pytest
    from simple_coder import product_of_list
    
    def test_product_of_list():
        assert product_of_list([2, 3, 4, 5]) == 120
        assert product_of_list([1, 2, 3, 4, 5]) == 120
        assert product_of_list([-2, 3, 4, -5]) == -120
        assert product_of_list([1, 1, 1, 1, 1]) == 1
        assert product_of_list([]) == 1
    
    def test_negative_numbers():
        with pytest.raises(ValueError):
            product_of_list([-2, 3, 4, 5])
    
    def test_empty_list():
        with pytest.raises(ValueError):
            product_of_list([])
    
    def test_non_numbers():
        with pytest.raises(TypeError):
            product_of_list([2, "three", 4, 5])
    
    def test_float_numbers():
        with pytest.raises(TypeError):
            product_of_list([2.5, 3.5, 4.5, 5.5])
    ```

    2024-03-01 21:02:39.592 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 136, completion_tokens: 316
    2024-03-01 21:02:39.595 | INFO     | metagpt.roles.role:_act:357 - Charlie(SimpleReviewer): to do SimpleWriteReview(SimpleWriteReview)
    

     Critical Comment: The test cases provided cover a variety of scenarios, including positive and negative numbers, empty lists, and non-number elements. However, they do not seem to cover the case where the input list contains only zeroes. It would be beneficial to add a test case for this scenario to ensure that the function behaves correctly when the product of the list is zero. Additionally, it would be useful to test the behavior of the function when the input list contains mixed types of numbers (e.g., integers and floats) to further validate its functionality.

    2024-03-01 21:02:45.925 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.004 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 422, completion_tokens: 115
    2024-03-01 21:02:45.928 | INFO     | __main__:_act:11 - Bob(SimpleTester): to do SimpleWriteTest(SimpleWriteTest)
    

     ```python
    import pytest
    from simple_coder import product_of_list
    
    def test_product_of_list():
        assert product_of_list([2, 3, 4, 5]) == 120
        assert product_of_list([1, 2, 3, 4, 5]) == 120
        assert product_of_list([-2, 3, 4, -5]) == -120
        assert product_of_list([1, 1, 1, 1, 1]) == 1
        assert product_of_list([]) == 1
    
    def test_zero_product():
        assert product_of_list([0, 0, 0, 0]) == 0
    
    def test_negative_numbers():
        with pytest.raises(ValueError):
            product_of_list([-2, 3, 4, 5])
    
    def test_empty_list():
        with pytest.raises(ValueError):
            product_of_list([])
    
    def test_non_numbers():
        with pytest.raises(TypeError):
            product_of_list([2, "three", 4, 5])
    
    def test_float_numbers():
        with pytest.raises(TypeError):
            product_of_list([2.5, 3.5, 4.5, 5.5])
    ```

    2024-03-01 21:03:04.995 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 569, completion_tokens: 348
    2024-03-01 21:03:04.998 | INFO     | metagpt.roles.role:_act:357 - Charlie(SimpleReviewer): to do SimpleWriteReview(SimpleWriteReview)
    

     Critical Comment: The test cases provided cover a variety of scenarios, including positive and negative numbers, empty lists, and non-number elements. However, they do not seem to cover the case where the input list contains only zeroes. It would be beneficial to add a test case for this scenario to ensure that the function behaves correctly when the product of the list is zero. Additionally, it would be useful to test the behavior of the function when the input list contains mixed types of numbers (e.g., integers and floats) to further validate its functionality.

    2024-03-01 21:03:11.699 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 887, completion_tokens: 115
    

## 2 多智能体案例：辩论


```python
import fire
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
```

### 2.1 定义动作


```python
class SpeakAloud(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name}, you are in a debate with {opponent_name}.
    ## DEBATE HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn, you should closely respond to your opponent's latest argument, state your position, 
    defend your arguments, and attack your opponent's arguments,
    craft a strong and emotional response in 80 words, in {name}'s rhetoric and viewpoints, your will argue:
    """
    name: str = "SpeakAloud"

    async def run(self, context: str, name: str, opponent_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, opponent_name=opponent_name)

        rsp = await self._aask(prompt)

        return rsp
```

### 2.2 定义角色


```python
class Debater(Role):
    name: str = ""
    profile: str = ""
    opponent_name: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self._init_actions([SpeakAloud])
        self._watch([UserRequirement, SpeakAloud])

    async def _observe(self, **kwargs) -> int:
        await super()._observe()
        # accept messages sent (from opponent) to self, disregard own messages from the last round
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # An instance of SpeakAloud

        memories = self.get_memories()
        context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        # print(context)

        rsp = await todo.run(context=context, name=self.name, opponent_name=self.opponent_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.opponent_name,
        )
        self.rc.memory.add(msg)

        return msg
```

### 2.3 实例化


```python
async def debate(idea: str, investment: float = 3.0, n_round: int = 5):
    """Run a team of presidents and watch they quarrel. :)"""
    Biden = Debater(name="Biden", profile="Democrat", opponent_name="Trump")
    Trump = Debater(name="Trump", profile="Republican", opponent_name="Biden")
    team = Team()
    team.hire([Biden, Trump])
    team.invest(investment)
    # send debate topic to Biden and let him speak first
    team.run_project(idea, send_to="Biden")  
    await team.run(n_round=n_round)
```


```python
idea = "The U.S. should commit more in climate change fighting"
await debate(idea)
```

    2024-03-01 21:51:30.698 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 21:51:30.699 | INFO     | metagpt.config:get_default_llm_provider_enum:126 - API: LLMProviderEnum.ZHIPUAI
    2024-03-01 21:51:30.700 | INFO     | metagpt.team:invest:86 - Investment: $3.0.
    2024-03-01 21:51:30.701 | INFO     | __main__:_act:18 - Biden(Democrat): to do SpeakAloud(SpeakAloud)
    

     "Donald, your denial of climate change is akin to burying your head in the sand. The science is clear, and the American people are demanding action. Our planet is on fire, and we cannot afford to wait any longer. Your fossil fuel allies may be padding your pockets, but they're burning our future. I believe in science, and I'll fight for a greener, more prosperous America. Together, we can tackle this crisis and secure our children's inheritance. Let's move forward, not backward."

    2024-03-01 21:51:36.691 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 139, completion_tokens: 109
    2024-03-01 21:51:36.694 | INFO     | __main__:_act:18 - Trump(Republican): to do SpeakAloud(SpeakAloud)
    

     "Joe, you're living in a fantasy world. The left-wing media may have brainwashed you into believing climate change is the end of the world, but I'm here to tell you the truth: it's not. Our economy depends on fossil fuels, and I won't let your socialist agenda destroy jobs and bankrupt America. I've repealed countless regulations, cutting red tape and unleashing American energy. Under my presidency, we've achieved record low unemployment and a booming economy.
    
    Your claims about science are laughable. I've met with top scientists and they agree that climate change is not a crisis. We need to focus on real problems, like border security and jobs. Instead of wasting money on green initiatives, we should invest in American workers and infrastructure. I'm standing up for common sense, and I'll continue to fight for the forgotten men and women of this country who deserve a better future."

    2024-03-01 21:51:47.293 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 248, completion_tokens: 197
    2024-03-01 21:51:47.295 | INFO     | __main__:_act:18 - Biden(Democrat): to do SpeakAloud(SpeakAloud)
    

     Biden: Donald, your dismissal of climate change is dangerous and irresponsible. The science is crystal clear, and our planet is paying the price for your denial. Our economy, our health, and our future are at stake. Yes, we need to protect American jobs, but investing in clean energy will create new opportunities and secure our prosperity. Your claim to have repealed regulations to boost the economy is a smoke screen: the fossil fuel industry received $20 billion in subsidies under your watch. I stand with the majority of Americans who believe in science and want a better future for our children. It's time to put our country before your personal interests and embrace a cleaner, more equitable future. Together, we can tackle climate change and build a stronger, more prosperous America.

    2024-03-01 21:51:55.975 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 485, completion_tokens: 158
    2024-03-01 21:51:55.978 | INFO     | __main__:_act:18 - Trump(Republican): to do SpeakAloud(SpeakAloud)
    

     "Joe, you're trying to dupe the American people with your doom-and-gloom climate predictions. I've spoken with real experts who disagree with your extremist views. Our economy thrives on fossil fuels, and I'm protecting American jobs by cutting red tape and promoting energy independence. Your claim about clean energy creating new opportunities is naive – wind and solar jobs can't compete with the prosperity generated by our robust energy sector. I've cut subsidies to fossil fuels, and instead invested in American workers and infrastructure.
    
    You talk about a cleaner, more equitable future, but your socialist agenda would bankrupt our country and destroy jobs. I'm standing up for common sense, and I won't let your fantasy climate policies dismantle our economy. We need to focus on real problems, like border security and jobs, not waste money on costly green initiatives. I'm fighting for the forgotten men and women who want a better future for their children – a future built on American energy and prosperity, not fantasy climate change policies."

    2024-03-01 21:52:07.459 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 650, completion_tokens: 216
    2024-03-01 21:52:07.462 | INFO     | __main__:_act:18 - Biden(Democrat): to do SpeakAloud(SpeakAloud)
    

     Biden: Donald, your stubborn denial of climate change is putting our planet, economy, and future at risk. You claim to stand for American jobs, but your failed policies have left us dependent on fossil fuels, exacerbating the climate crisis. The science is clear – we must invest in clean energy to secure our prosperity and protect our children's inheritance.
    
    Your assertion that clean energy can't compete with fossil fuels is outdated and false. Renewable energy sources are rapidly evolving, and countries like China and Germany are already surpassing us in this sector. We can create new jobs, reduce greenhouse gas emissions, and lead the global transition to a cleaner economy.
    
    Your scaremongering about socialism and bankrupting our country is a cynical ploy to distract from your own record of enriching the wealthy and neglecting the needs of ordinary Americans. I'm fighting for justice and a better future for all, where we prioritize people over profits and protect our planet.
    
    The American people are tired of your empty promises and denial of the obvious. It's time to face the facts, take responsibility, and join us in building a greener, more equitable America. Together, we can tackle climate change and ensure a brighter future for all.

    2024-03-01 21:52:21.411 | INFO     | metagpt.utils.cost_manager:update_cost:48 - Total running cost: $0.005 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 916, completion_tokens: 256
    
