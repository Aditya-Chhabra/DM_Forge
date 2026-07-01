from dataclasses import dataclass, field
from typing import Any, Dict
import json
import re

from llm.router import LLMRouter


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _run_llm(
    router: LLMRouter,
    prompt: str,
    system_prompt: str = "You are a precise structured JSON generator.",
    temperature: float = 0.4,
    max_tokens: int = 800,
) -> str:
    return router.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    ).text


def _safe_json(text: str) -> dict:
    if not text:
        return {}
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def research_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    router: LLMRouter = context["router"]
    text: str = context.get("input_text", "").strip()
    if not text:
        return {"research": {"error": "empty_input"}}

    prompt = f"""You are an expert LinkedIn content analyst. Your job is to deeply understand what this post is about and what the BEST response approach would be.

## YOUR TASK

Analyze the following LinkedIn post and determine:

1. **What is the author trying to achieve?** (sharing knowledge, asking for help, celebrating, hiring, seeking opinions, starting discussion, etc.)

2. **Does this post contain a question or request?** If yes, extract it exactly. A good response should ADDRESS this, not ignore it.

3. **What would make the author WANT to reply?** Think about what response would genuinely interest them vs what would feel generic/spammy.

4. **What specific details in the post can be referenced?** (names, numbers, concepts, achievements, challenges mentioned)

5. **What is the ideal response type?**
   - If author asks a question → provide a thoughtful answer or share relevant experience
   - If author shares achievement → acknowledge specifically what they achieved, then explore
   - If author shares knowledge/tip → show you understood it, ask about edge cases or their experience
   - If author is hiring → show genuine interest in specific aspects of the role/company
   - If author seeks opinions → share your perspective with reasoning
   - If author shares a struggle → empathize and offer insight if you have relevant experience

## THINK STEP BY STEP

Before generating output, reason through:
- What does the author explicitly want? (stated or implied)
- What would feel like a genuine, valuable response to them?
- What would feel like spam or a generic template?
- If I were this author, what reply would make me want to respond back?

## OUTPUT (strict JSON)

{{
  "analysis": {{
    "post_summary": "1-2 sentence summary of what this post is about",
    "author_goal": "what the author wants to achieve with this post",
    "explicit_question_or_request": "exact question/request if present, null if none",
    "key_details": ["specific detail 1", "specific detail 2", "specific detail 3"],
    "emotional_tone": "celebratory/frustrated/curious/informative/seeking_help/etc"
  }},
  
  "response_strategy": {{
    "ideal_response_type": "answer_their_question | acknowledge_and_explore | share_perspective | show_interest | empathize_and_relate",
    "what_to_lead_with": "what should the first sentence accomplish",
    "what_value_to_provide": "what makes this response worth reading for the author",
    "should_ask_question": true/false,
    "question_purpose": "why ask this question / what it achieves (or null if no question needed)",
    "avoid": "what would make this response feel generic or spammy"
  }},
  
  "hooks": [
    "contextual opening line 1 based on strategy",
    "contextual opening line 2 based on strategy",
    "contextual opening line 3 based on strategy"
  ],
  
  "reasoning": "brief explanation of why this strategy fits this specific post"
}}

---

POST TO ANALYZE:

{text}"""

    raw = _run_llm(
        router, 
        prompt,
        system_prompt="You are an expert at understanding social context and crafting genuine, valuable responses. Think deeply before responding.",
        temperature=0.4,
        max_tokens=900
    )
    data = _safe_json(raw)
    
    if not data:
        data = {}
    
    return {"research": data}


def strategy_planner_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    router: LLMRouter = context["router"]
    text = context.get("input_text", "")
    research = context.get("research", {})
    
    prompt = f"""You are a master strategist for crafting LinkedIn DMs that get genuine responses.

## CONTEXT

You have research analysis of a LinkedIn post. Now you must plan the EXACT strategy for the DM.

## YOUR TASK

Based on the research, decide:

1. **Response Mode**: What type of response fits best?
   - ANSWER MODE: If the post asks a question or seeks advice, lead with your answer/insight
   - ACKNOWLEDGE MODE: If the post shares an achievement or news, acknowledge specifically first
   - RELATE MODE: If the post shares an experience/struggle, relate with your own experience
   - CURIOUS MODE: If the post shares knowledge, show understanding and ask to go deeper
   - ENGAGE MODE: If the post seeks opinions/debate, share your perspective

2. **Content Balance**: What's the right mix?
   - How much should you GIVE (answer, insight, perspective, acknowledgment)?
   - How much should you ASK (questions, clarifications)?
   - If the author asked something, answering should come BEFORE asking anything new

3. **Specificity Level**: How detailed should the DM be?
   - What specific details from the post MUST be referenced to avoid feeling generic?
   - What would prove you actually read and understood the post?

4. **Question Strategy** (if asking any questions):
   - Should you ask a question at all? (not always necessary)
   - If yes, what's the PURPOSE of the question?
   - Is it to learn something, to show interest, or to start dialogue?

## THINK THROUGH

- What would make this author WANT to reply?
- What would make them think "this person actually gets it"?
- What would make them think "this is just a template"?

## OUTPUT (strict JSON)

{{
  "response_mode": "answer | acknowledge | relate | curious | engage",
  
  "content_plan": {{
    "lead_with": "exactly what the first 1-2 sentences should accomplish",
    "provide_value_by": "what insight/acknowledgment/perspective to include",
    "reference_these_details": ["detail1", "detail2"],
    "end_with": "how to close the message"
  }},
  
  "question_decision": {{
    "should_ask_question": true/false,
    "why_or_why_not": "reasoning",
    "if_yes_question_goal": "what the question achieves",
    "question_type": "follow_up | clarifying | exploratory | none"
  }},
  
  "tone": "warm | professional | casual | thoughtful",
  
  "length_guidance": "short (2-3 sentences) | medium (3-4 sentences) | detailed (4-5 sentences)",
  
  "must_avoid": ["things that would make this feel generic or spammy"],
  
  "success_criteria": "what would make this DM successful"
}}

---

POST:
{text}

RESEARCH ANALYSIS:
{json.dumps(research, indent=2) if isinstance(research, dict) else research}"""

    raw = _run_llm(
        router, 
        prompt,
        system_prompt="You are a strategist who understands human psychology and what makes people want to engage in conversation.",
        temperature=0.3,
        max_tokens=700
    )
    data = _safe_json(raw)
    
    return {"strategy": data if data else {}}


def writer_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    router: LLMRouter = context["router"]
    text = context.get("input_text", "")
    research = context.get("research", {})
    strategy = context.get("strategy", {})
    addressee_name = _single_line(context.get("addressee_name", ""))
    
    prompt = f"""You are a skilled communicator writing a LinkedIn DM that will genuinely connect with the author.

## CONTEXT

You have:
1. The original post
2. Deep analysis of what the post is about and what the author wants
3. A strategy for how to respond

Your job: Write the actual DM following the strategy, making it feel human and genuine.

## CRITICAL RULES

1. **Be Specific**: Reference actual details from the post. Never write something that could apply to any post.

2. **Match the Response Mode**:
   - If strategy says "answer" → Lead with your answer/insight to their question
   - If strategy says "acknowledge" → Lead with specific acknowledgment of what they achieved/shared
   - If strategy says "relate" → Lead with how their experience connects to yours
   - If strategy says "curious" → Show you understood their point, then ask to go deeper
   - If strategy says "engage" → Share your perspective on their topic

3. **Question Logic**:
   - If the post ASKED a question → Your primary job is to ANSWER it, not ask another question
   - Only ask a question if strategy indicates you should, and make sure the question has a clear purpose
   - Questions should feel like natural curiosity, not an interview

4. **Length**: 40-90 words. Concise but substantive.

5. **Tone**: Match the strategy's tone guidance. Sound like a real person, not a sales bot.

6. **Opening**: If you have their name, use it naturally. If not, dive straight into the content (no awkward "Hi there").

## WHAT TO AVOID

- Generic phrases like "Great post!", "Love this!", "Thanks for sharing!"
- Questions that don't connect to something specific in the post
- Sounding like you're trying to sell something
- Being so formal it feels robotic
- Asking questions when you should be providing value first

## ADDRESSEE NAME (if available): {addressee_name if addressee_name else "Not provided - skip name greeting"}

## OUTPUT (strict JSON)

{{
  "message": "the complete DM message",
  "why_this_works": "1 sentence explaining why this message fits the context",
  "key_elements": ["what specific details from post were referenced"]
}}

---

ORIGINAL POST:
{text}

RESEARCH ANALYSIS:
{json.dumps(research, indent=2) if isinstance(research, dict) else research}

STRATEGY:
{json.dumps(strategy, indent=2) if isinstance(strategy, dict) else strategy}"""

    raw = _run_llm(
        router, 
        prompt,
        system_prompt="You write like a thoughtful human, not a corporate bot. Be genuine, specific, and valuable.",
        temperature=0.5,
        max_tokens=500
    )
    data = _safe_json(raw)
    
    message = _single_line(data.get("message", ""))
    if not message:
        hooks = research.get("hooks", [])
        if hooks and isinstance(hooks, list) and hooks[0]:
            message = _single_line(hooks[0])
        else:
            message = "Interesting perspective - what led you to this approach?"
    
    return {"writer": {"message": message, "analysis": data}}

def editor_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    router: LLMRouter = context["router"]
    text = context.get("input_text", "")
    research = context.get("research", {})
    strategy = context.get("strategy", {})
    writer = context.get("writer", {})
    message = _single_line(writer.get("message", ""))
    addressee_name = _single_line(context.get("addressee_name", ""))

    prompt = f"""You are a quality reviewer ensuring this LinkedIn DM achieves its goal.

## YOUR TASK

Review the draft DM against the original post and strategy. Check if the message:
1. Actually addresses what the post was about
2. Follows the planned strategy (answer if should answer, acknowledge if should acknowledge, etc.)
3. References specific details from the post (not generic)
4. Sounds human and genuine
5. Would make the author want to respond

## REVIEW CRITERIA

**Response Appropriateness:**
- If the post asked a question: Does the DM provide an answer or relevant insight? (It should!)
- If the post shared an achievement: Does the DM acknowledge it specifically? (It should!)
- If the strategy said to provide value first: Does it?

**Specificity Check:**
- Does the DM mention something specific from the post that proves it was actually read?
- Could this message be sent to ANY post, or is it clearly tailored?

**Tone & Flow:**
- Does it sound like a real person wrote it?
- Is it free of corporate buzzwords and sales-y language?
- Does it flow naturally?

**Question Quality (if there's a question):**
- Is the question connected to something in the post?
- Does it have a clear purpose?
- If the post asked a question, did we answer BEFORE asking our own?

## YOUR JOB

1. Identify any issues
2. If issues exist, fix them minimally (don't rewrite unnecessarily)
3. If the message is good, keep it as-is
4. Add the addressee name naturally if provided and not already included

## ADDRESSEE NAME: {addressee_name if addressee_name else "Not provided"}

## OUTPUT (strict JSON)

{{
  "final_message": "the final DM (fixed if needed, or original if good)",
  "quality_score": 1-10,
  "issues_found": ["list of issues if any"],
  "changes_made": ["what was fixed, or 'none' if message was kept as-is"],
  "why_this_works": "brief explanation of why the final message fits the context"
}}

---

ORIGINAL POST:
{text}

RESEARCH ANALYSIS:
{json.dumps(research, indent=2) if isinstance(research, dict) else research}

STRATEGY:
{json.dumps(strategy, indent=2) if isinstance(strategy, dict) else strategy}

DRAFT MESSAGE TO REVIEW:
{message}"""

    raw = _run_llm(
        router,
        prompt=prompt,
        system_prompt="You ensure messages are genuine, specific, and achieve their communication goal.",
        temperature=0.2,
        max_tokens=500,
    )
    data = _safe_json(raw)

    final_message = _single_line(data.get("final_message", message))
    
    if not final_message:
        final_message = message if message else "Interesting perspective - I'd love to hear more about your approach."
    
    if addressee_name and addressee_name.lower() not in final_message.lower():
        final_message = _enforce_named_greeting(final_message, addressee_name)
    
    score = data.get("quality_score", 0)
    try:
        score_value = float(score)
    except Exception:
        score_value = 0.0
    score_value = max(0.0, min(10.0, round(score_value, 2)))

    return {
        "editor": {
            "final_message": final_message,
            "score": score_value,
            "review": data,
        }
    }


def _enforce_named_greeting(message: str, addressee_name: str) -> str:
    clean_message = _single_line(message)
    name = _single_line(addressee_name)
    if not name:
        return clean_message
    if clean_message.lower().startswith(("hi ", "hey ", "hello ")):
        return re.sub(
            r"^(hi|hey|hello)\s*,?\s*[^,]*,?",
            f"Hi {name},",
            clean_message,
            flags=re.IGNORECASE,
            count=1,
        ).strip()
    return f"Hi {name}, {clean_message}"


@dataclass
class PipelineResult:
    post: str
    hook: str
    draft_dm: str
    edited_dm: str
    final_dm: str
    research: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)


class CrewPipeline:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.steps = [
            research_agent,
            strategy_planner_agent,
            writer_agent,
            editor_agent,
        ]

    def seed_demo_cache(
        self,
        post_text: str,
        hook: str,
        draft_dm: str = "",
        final_dm: str = "",
    ) -> None:
        research_response = json.dumps({
            "analysis": {
                "post_summary": "Demo post analysis",
                "author_goal": "sharing",
                "explicit_question_or_request": None,
                "key_details": [],
                "emotional_tone": "informative"
            },
            "response_strategy": {
                "ideal_response_type": "curious",
                "what_to_lead_with": hook,
                "what_value_to_provide": "genuine engagement",
                "should_ask_question": True,
                "question_purpose": "learn more",
                "avoid": "generic phrases"
            },
            "hooks": [hook],
            "reasoning": "Demo seeded response"
        })
        
        research_prompt_fragment = f"POST TO ANALYZE:\n\n{post_text}"
        
        self.router.set_cached(
            self.router._cache_key(
                prompt=research_prompt_fragment,
                system_prompt="You are an expert at understanding social context and crafting genuine, valuable responses. Think deeply before responding.",
                temperature=0.4,
                max_tokens=900,
                sensitive=False,
            ),
            research_response,
            "demo",
        )

    def _run_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(context)
        current["router"] = self.router
        for step in self.steps:
            result = step(current)
            if isinstance(result, dict):
                current.update(result)
        return current

    def run(self, input_data: str | Dict[str, Any]) -> PipelineResult:
        if isinstance(input_data, str):
            run_context: Dict[str, Any] = {"input_text": input_data}
        else:
            run_context = dict(input_data)
        context = self._run_context(run_context)
        
        research = context.get("research", {})
        strategy = context.get("strategy", {})
        writer = context.get("writer", {})
        editor = context.get("editor", {})
        
        hooks = research.get("hooks", [])
        hook = hooks[0] if hooks and isinstance(hooks, list) else ""
        
        draft = writer.get("message", "") if isinstance(writer, dict) else ""
        final = editor.get("final_message", draft) if isinstance(editor, dict) else draft
        post_text = run_context.get("input_text", "") if isinstance(run_context, dict) else ""
        
        return PipelineResult(
            post=post_text,
            hook=hook,
            draft_dm=draft,
            edited_dm=final,
            final_dm=final,
            research=research,
            strategy=strategy,
        )
