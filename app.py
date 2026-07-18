import gradio as gr

from inference import generate_response


def chat(question):

    if not question.strip():
        return "Please enter an HR-related question."

    return generate_response(question)


demo = gr.Interface(

    fn=chat,

    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask an HR policy question...",
        label="Question",
    ),

    outputs=gr.Textbox(
        lines=10,
        label="Answer",
    ),

    title="HR Policy Assistant",

    description="""
Ask questions about company HR policies.
The responses are generated using a fine-tuned Large Language Model.
""",

    allow_flagging="never",

)


if __name__ == "__main__":
    demo.launch()
