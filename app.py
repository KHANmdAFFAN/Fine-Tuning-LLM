"""
app.py

Gradio interface for the HR Policy Assistant.
"""

import gradio as gr

from inference import generate_response


def chat(question, context):
    if not question.strip():
        return "Please enter a question."

    return generate_response(
        question=question,
        input_text=context,
    )


demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(
            label="HR Question",
            placeholder="Example: What is the leave policy?"
        ),
        gr.Textbox(
            label="Additional Context (Optional)",
            placeholder="Department, employee type, etc."
        ),
    ],
    outputs=gr.Textbox(
        label="HR Policy Assistant"
    ),
    title="HR Policy Assistant",
    description="""
Ask any HR policy related question.

Examples:
• What is the maternity leave policy?
• Can I carry forward annual leave?
• What happens during probation?
""",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
