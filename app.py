from crewai import Agent, Task, Crew, LLM
import gradio as gr

def criar_crew(tema):
    
    groq_llm = LLM(model="groq/llama-3.3-70b-versatile")

    pesquisador = Agent(
        role="Pesquisador",
        goal="Buscar informações detalhadas sobre o assunto informado.",
        backstory="Um especialista em coleta de dados e análise de informações.",
        llm=groq_llm
    )

    redator = Agent(
        role="Redator",
        goal="Resumir as informações encontradas e criar um relatório.",
        backstory="Um redator experiente que transforma dados em textos claros.",
        llm=groq_llm
    )

    latex = Agent(
        role="Latex",
        goal="Crie um documento Latex.",
        backstory="Um criador de documentos Latex.",
        llm=groq_llm
    )

    tarefa_pesquisa = Task(
        description="Pesquise informações detalhadas sobre {}.".format(tema),
        agent=pesquisador,
        expected_output='Informações corretas e detalhadas sobre o assunto informado'
    )

    tarefa_redacao = Task(
        description="Com base nas informações fornecidas pelo pesquisador, crie um resumo claro e objetivo.",
        agent=redator,
        expected_output='Resumo claro e objetivo'
    )

    tarefa_latex = Task(
        description="Com base no resumo, crie um documento latex.",
        agent=latex,
        expected_output='Um documento Latex'
    )

    equipe = Crew(
        agents=[pesquisador, redator, latex],
        tasks=[tarefa_pesquisa, tarefa_redacao, tarefa_latex]
    )
    
    return equipe

def executar_crew(entrada):
    equipe = criar_crew(entrada)
    resultado = equipe.kickoff()
    return resultado

# Interface Gradio
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Agentes: Pesquisa e Relatório em Latex")
            gr.Markdown("Clique no botão para executar a equipe de agentes (três) que irão pesquisar e criar um relatório sobre o assunto informados.")
            tema_input = gr.Textbox(label="Informe o tema")
            submit_button = gr.Button(value="Executar")
        with gr.Column():
            gr.Image(value="diagrama.png", label="Arquitetura interna dos agentes", width=400)
    with gr.Row():
        output_text = gr.Textbox(label="Saída documento no formato Latex", show_copy_button=True)    
    
    submit_button.click(fn=executar_crew, 
                        inputs=[tema_input],
                        outputs=output_text)
    
if __name__ == "__main__":
    interface.launch()
