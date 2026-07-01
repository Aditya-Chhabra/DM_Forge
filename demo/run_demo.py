from crew import CrewPipeline
from demo.fixtures import DEMO_CASES
from llm import LLMRouter


def run_demo() -> list[dict[str, str]]:
    router = LLMRouter()
    pipeline = CrewPipeline(router)
    for case in DEMO_CASES:
        pipeline.seed_demo_cache(
            post_text=case["post"],
            hook=case["hook"],
            draft_dm=case["draft_dm"],
            final_dm=case["final_dm"],
        )

    outputs: list[dict[str, str]] = []
    for case in DEMO_CASES:
        result = pipeline.run(case["post"])
        outputs.append(
            {
                "name": case["name"],
                "input_post": case["post"],
                "hook": result.hook,
                "dm": result.final_dm,
            }
        )
    return outputs


if __name__ == "__main__":
    rows = run_demo()
    for row in rows:
        print(f"\nCase: {row['name']}")
        print(f"Hook: {row['hook']}")
        print(f"DM: {row['dm']}")
