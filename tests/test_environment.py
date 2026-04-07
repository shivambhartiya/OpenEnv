from nl_reward_env.baseline import BaselineAgent
from nl_reward_env.environment import NaturalLanguageRewardEnvironment
from nl_reward_env.models import NaturalLanguageRewardAction
from nl_reward_env.tasks import list_tasks


def test_environment_supports_all_tasks():
    env = NaturalLanguageRewardEnvironment()
    agent = BaselineAgent()

    for task in list_tasks():
        obs = env.reset(task_id=task.task_id)
        assert obs.task_id == task.task_id
        action = agent._fallback_action(obs)
        next_obs = env.step(NaturalLanguageRewardAction(response=action))
        assert 0.0 <= float(next_obs.reward or 0.0) <= 1.0
        assert env.state.task_id == task.task_id
        assert env.state.step_count == 1
