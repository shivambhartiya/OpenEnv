"""Baseline policy used by inference.py."""

from __future__ import annotations

import json

from openai import OpenAI

from nl_reward_env.config import RuntimeConfig, load_runtime_config
from nl_reward_env.models import NaturalLanguageRewardObservation


class BaselineAgent:
    """A practical baseline with OpenAI-backed generation and safe fallbacks."""

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or load_runtime_config()
        self._client = (
            OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )
            if self.config.api_key and self.config.model_name
            else None
        )

    def act(self, observation: NaturalLanguageRewardObservation, step: int) -> str:
        if self._client is None or not self.config.model_name:
            return self._fallback_action(observation)

        prompt = {
            "task_name": observation.task_name,
            "difficulty": observation.difficulty,
            "instruction": observation.instruction,
            "scenario": observation.scenario,
            "response_format": observation.response_format,
            "last_feedback": observation.last_feedback,
            "last_submission": observation.last_submission,
            "step": step,
            "messages_remaining": observation.messages_remaining,
        }
        try:
            completion = self._client.chat.completions.create(
                model=self.config.model_name,
                temperature=self.config.default_temperature,
                max_completion_tokens=500,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a capable RL baseline policy. Produce only the "
                            "next action text in the exact format requested by the task."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=True, indent=2),
                    },
                ],
            )
            content = (completion.choices[0].message.content or "").strip()
            return content or self._fallback_action(observation)
        except Exception:
            return self._fallback_action(observation)

    def _fallback_action(self, observation: NaturalLanguageRewardObservation) -> str:
        if observation.task_id == "customer_support_response":
            return (
                "I am sorry your order has been delayed, and I understand why this is "
                "so frustrating with the birthday coming up. I cannot guarantee delivery "
                "today, but I can open an urgent trace with our logistics team right now "
                "and review the latest tracking updates for order #48152. If you reply "
                "with the best shipping address and contact number, I will confirm the "
                "details on the case and follow up as quickly as possible. If the package "
                "has been in transit for 5 business days, we can also review refund "
                "eligibility, and if the trace confirms a delivery failure sooner we can "
                "look at an expedited replacement."
            )
        if observation.task_id == "email_triage":
            return json.dumps(
                {
                    "priority": "high",
                    "category": "security_fraud",
                    "assignee": "security@acme.co",
                    "justification": (
                        "An external sender is requesting a same-day bank detail change "
                        "for a wire, which matches business-email-compromise risk and "
                        "could cause financial loss."
                    ),
                    "response_draft": (
                        "Thanks for the note. We cannot process payment-account changes "
                        "from email alone. The request has been routed for security "
                        "review, and any update would require out-of-band verification "
                        "through a trusted company channel."
                    ),
                }
            )
        return (
            "- Severity: High | Lines: 5-6 | Issue: This lets a regular authenticated "
            "user set `payload[\"role\"]` directly, which creates a privilege-escalation "
            "path on a self-service endpoint.\n"
            "- Severity: High | Lines: 5-6 | Fix: Add a server-side authorization check "
            "before accepting role changes, and allowlist permitted role transitions for "
            "admin-only callers."
        )
