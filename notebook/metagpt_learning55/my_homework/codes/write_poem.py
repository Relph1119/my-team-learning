#!/usr/bin/env python
# encoding: utf-8
"""
@file: write_poem.py
@time: 2024/3/1 19:35
@project: my-team-learning
@desc: 学生写一遍优美的诗歌，老师进行评审，直到符合老师的要求。
"""

import asyncio

from metagpt.actions import Action, UserRequirement
from metagpt.const import MESSAGE_ROUTE_TO_ALL
from metagpt.environment import Environment
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

classroom = Environment()


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


class ReviewPoem(Action):
    name: str = "ReviewPoem"

    PROMPT_TEMPLATE: str = """

    Here is the historical conversation record : {msg} .
    Check student-created poems about the subject provided by human and give your suggestions for revisions. 
    You prefer poems with elegant sentences and retro style.
    Return only your comments with NO other texts.
    your comments:
    """

    async def run(self, msg: str):
        prompt = self.PROMPT_TEMPLATE.format(msg=msg)

        rsp = await self._aask(prompt)

        return rsp


class Student(Role):
    name: str = "xiaoming"
    profile: str = "Student"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([WritePoem])
        self._watch([UserRequirement, ReviewPoem])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        todo = self.rc.todo

        msg = self.get_memories()  # 获取所有记忆
        # logger.info(msg)
        poem_text = await WritePoem().run(msg)
        logger.info(f'student : {poem_text}')
        msg = Message(content=poem_text, role=self.profile,
                      cause_by=type(todo))

        return msg


class Teacher(Role):
    name: str = "laowang"
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


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main(topic='write a poem about moon'))
