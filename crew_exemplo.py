from crewai import Agent, Task, Crew
from langchain.llms import OpenAI

pesquisador = Agent(
    role="Pesquisador",
    goal="Buscar informações detalhadas sobre um tópico.",
    backstory="Um especialista em coleta de dados e análise de informações.",
    llm=OpenAI(model_name="gpt-4o-mini", temperature=0.7)
)

redator = Agent(
    role="Redator",
    goal="Resumir as informações encontradas e criar um relatório.",
    backstory="Um redator experiente que transforma dados em textos claros.",
    llm=OpenAI(model_name="gpt-4o-mini", temperature=0.5)
)

latex = Agent(
    role="Latex",
    goal="Crie um documento Latex.",
    backstory="Um criador de documentos Latex.",
    llm=OpenAI(model_name="gpt-4o-mini", temperature=0.5)
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

# Criando a equipe (Crew)
equipe = Crew(
    agents=[pesquisador, redator],
    tasks=[tarefa_pesquisa, tarefa_redacao, tarefa_latex]
)

# Executando a equipe
resultado = equipe.kickoff()
print(resultado)
