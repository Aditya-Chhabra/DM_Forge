from crew import CrewPipeline
from input import InputAdapter
from llm import LLMRouter

import gradio as gr


def _build_pipeline() -> tuple[InputAdapter, CrewPipeline]:
    router = LLMRouter()
    adapter = InputAdapter()
    pipeline = CrewPipeline(router=router)
    return adapter, pipeline


def build_app() -> gr.Blocks:
    adapter, pipeline = _build_pipeline()

    def generate_dm(raw_input: str) -> str:
        post = adapter.resolve_post(raw_input)
        addressee_name = adapter.extract_addressee_name(raw_input)
        result = pipeline.run({"input_text": post, "addressee_name": addressee_name})
        if hasattr(result, "final_dm"):
            return result.final_dm
        if isinstance(result, dict):
            editor = result.get("editor", {})
            if isinstance(editor, dict):
                return editor.get("final_message", "")
        return ""

    with gr.Blocks(title="DM Forge") as app:
        gr.Markdown("# DM Forge")
        gr.Markdown(
            "Paste a LinkedIn post or public LinkedIn profile URL. "
            "The app auto-infers context and creates a draft DM for manual use only."
        )
        user_input = gr.Textbox(
            label="LinkedIn Post or Profile URL",
            placeholder="Paste post text or https://www.linkedin.com/in/username/",
            lines=6,
        )
        generate_btn = gr.Button("Generate DM")
        dm_output = gr.Textbox(label="Generated LinkedIn DM", lines=4)
        disclaimer = (
            "Disclaimer: This tool only uses public information and generates draft messages. "
            "It does not send LinkedIn messages automatically."
        )
        gr.Markdown(disclaimer)

        generate_btn.click(fn=generate_dm, inputs=user_input, outputs=dm_output)

    return app


def launch_app() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    launch_app()
