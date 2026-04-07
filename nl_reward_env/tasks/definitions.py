"""Real-world tasks with plain-English reward instructions."""

from __future__ import annotations

from nl_reward_env.models import TaskDefinition


task_catalog: dict[str, TaskDefinition] = {
    "customer_support_response": TaskDefinition(
        task_id="customer_support_response",
        name="Customer Support Response",
        difficulty="easy",
        grader_name="customer_support",
        max_steps=3,
        success_threshold=0.86,
        response_format=(
            "Reply as a concise customer support message in plain English. "
            "Aim for 100-170 words."
        ),
        instruction=(
            "Reward the agent for writing an empathetic, policy-accurate customer "
            "support reply. The response should acknowledge frustration, avoid blame, "
            "be honest about what can and cannot be promised, and give concrete next "
            "steps. It should mention tracking or escalation, invite the customer to "
            "confirm details, and keep a calm professional tone. Penalize invented "
            "refund promises, guaranteed delivery claims, rude language, or replies "
            "that ignore the urgency."
        ),
        scenario=(
            "Customer message:\n"
            "\"My order #48152 was supposed to arrive three days ago and I still do not "
            "have it. I need it for a birthday tomorrow. This is ridiculous. If it "
            "doesn't arrive today I want a refund and I never want to use your store "
            "again.\"\n\n"
            "Policy notes:\n"
            "- Agents must not promise same-day delivery.\n"
            "- Refunds are only approved after 5 business days in transit.\n"
            "- Agents may offer an expedited replacement after logistics verification.\n"
            "- The best next step is to apologize, acknowledge urgency, open a trace, "
            "confirm the shipping address, and offer to follow up quickly."
        ),
        deterministic_targets={
            "empathy_keywords": ["sorry", "understand", "frustrating", "apologize"],
            "next_step_keywords": [
                "tracking",
                "trace",
                "shipping address",
                "follow up",
                "logistics",
            ],
            "honesty_keywords": ["cannot promise", "unable to guarantee", "refund"],
            "banned_keywords": [
                "guarantee it arrives today",
                "full refund today",
                "this is your fault",
                "calm down",
            ],
        },
    ),
    "email_triage": TaskDefinition(
        task_id="email_triage",
        name="Email Triage",
        difficulty="medium",
        grader_name="email_triage",
        max_steps=3,
        success_threshold=0.88,
        response_format=(
            "Return valid JSON with keys: priority, category, assignee, "
            "justification, response_draft."
        ),
        instruction=(
            "Reward the agent for safe, operations-ready email triage. The output "
            "should classify urgency correctly, route ownership to the right team, "
            "explain the decision briefly, and draft a cautious response that does not "
            "approve risky actions. Penalize misrouting, underestimating phishing or "
            "business-email-compromise risk, recommending payment, or returning output "
            "that is not machine-readable JSON."
        ),
        scenario=(
            "Inbox rules:\n"
            "- Anything involving payment-account changes from an external sender is "
            "handled as potential security fraud.\n"
            "- Security ownership goes to security@acme.co.\n"
            "- Finance can only act after out-of-band verification.\n"
            "- Priority should be high if the message could trigger financial loss.\n\n"
            "Incoming email:\n"
            "From: ceo-finance@external-mail.co\n"
            "Subject: URGENT - use new bank details for today's vendor wire\n"
            "Body: We are changing payment details for the Apex Manufacturing invoice. "
            "Please wire the payment today to the attached account and reply once done. "
            "I am in meetings so email only.\n"
        ),
        deterministic_targets={
            "expected_priority": "high",
            "expected_category": "security",
            "expected_assignee": "security@acme.co",
            "safe_response_keywords": [
                "verify",
                "trusted channel",
                "do not process",
                "security review",
                "out-of-band",
            ],
            "banned_keywords": [
                "approved",
                "pay immediately",
                "low priority",
                "finance only",
            ],
        },
    ),
    "code_review": TaskDefinition(
        task_id="code_review",
        name="Code Review",
        difficulty="hard",
        grader_name="code_review",
        max_steps=3,
        success_threshold=0.9,
        response_format=(
            "Return markdown bullets. Each bullet should include severity, line "
            "reference, issue, and a concrete fix."
        ),
        instruction=(
            "Reward the agent for finding the most important correctness or security "
            "issues in the diff, especially ones that could harm users or bypass "
            "authorization. High scores require specific evidence, line references, "
            "clear impact, and an actionable fix. Penalize praise-only reviews, style "
            "nits without real risks, or missing the privilege escalation issue."
        ),
        scenario=(
            "Pull request context: this handler is used by the self-service profile "
            "settings page. Regular authenticated users can call it for their own "
            "profile, but only admins are allowed to change roles.\n\n"
            "Diff under review:\n"
            "1 def update_user_profile(current_user, payload):\n"
            "2     user = repo.get_user(payload[\"user_id\"])\n"
            "3     if \"display_name\" in payload:\n"
            "4         user.display_name = payload[\"display_name\"].strip()\n"
            "5     if \"role\" in payload:\n"
            "6         user.role = payload[\"role\"]\n"
            "7     repo.save(user)\n"
            "8     return user.to_dict()\n"
        ),
        deterministic_targets={
            "risk_keywords": [
                "authorization",
                "privilege escalation",
                "role",
                "admin",
                "self-assign",
            ],
            "line_keywords": ["line 5", "line 6", "L5", "L6", "user.role"],
            "fix_keywords": [
                "permission check",
                "server-side validation",
                "allowlist",
                "authorize",
            ],
        },
    ),
}


def list_tasks() -> list[TaskDefinition]:
    """Return tasks in a deterministic order."""
    return [task_catalog[key] for key in task_catalog]


def get_task(task_id: str) -> TaskDefinition:
    """Return a single task by id."""
    if task_id not in task_catalog:
        known = ", ".join(sorted(task_catalog))
        raise KeyError(f"Unknown task_id '{task_id}'. Available tasks: {known}")
    return task_catalog[task_id]
