from crewai import Agent, Task, Crew
from langchain_community.llms import OpenAI
import gradio as gr

def criar_crew():
    pesquisador = Agent(
        role="Pesquisador",
        goal="Buscar informações detalhadas sobre um tópico.",
        backstory="Um especialista em coleta de dados e análise de informações.",
        llm=OpenAI(model_name="gpt-4o-mini")
    )

    redator = Agent(
        role="Redator",
        goal="Resumir as informações encontradas e criar um relatório.",
        backstory="Um redator experiente que transforma dados em textos claros.",
        llm=OpenAI(model_name="gpt-4o-mini")
    )

    latex = Agent(
        role="Latex",
        goal="Crie um documento Latex.",
        backstory="Um criador de documentos Latex.",
        llm=OpenAI(model_name="gpt-4o-mini")
    )

    tarefa_pesquisa = Task(
        description="Pesquise informações detalhadas sobre Aprendizagem de Máquina na estimativa de esforço.",
        agent=pesquisador,
        expected_output='Informações corretas e detalhadas sobre aprendizagem de máquina na estimativa de esforço'
    )

    tarefa_redacao = Task(
        description="Com base nas informações fornecidas pelo pesquisador, crie um resumo claro e objetivo.",
        agent=redator,
        expected_output='Resumo claro e objetivo'
    )

    tarefa_latex = Task(
        description="Com base nas informações fornecidas pelo pesquisador, crie um documento latex.",
        agent=latex,
        expected_output='Um documento Latex'
    )

    equipe = Crew(
        agents=[pesquisador, redator],
        tasks=[tarefa_pesquisa, tarefa_redacao, tarefa_latex]
    )
    
    return equipe

def executar_crew():
    equipe = criar_crew()
    resultado = equipe.kickoff()
    return resultado

# Interface Gradio
interface = gr.Interface(
    fn=executar_crew,
    outputs="text",
    title="Crew AI - Pesquisa e Relatório em Latex",
    description="Clique no botão para executar a equipe de agentes que irão pesquisar e criar um relatório sobre Aprendizagem de Máquina em latex."
)

if __name__ == "__main__":
    interface.launch()
