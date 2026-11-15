import re

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
import unicodedata
import plots
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
from scipy.stats import norm
import statsmodels.stats.weightstats as ws
from scipy import stats
import requests
import io

versao_streamlit = tuple(int(parte) for parte in st.__version__.split(".")[:2])
LARGURA = {"width": "stretch"} if versao_streamlit >= (1, 50) else {"use_container_width": True}

st.set_page_config(
    page_title="IA003 — Trabalho Final - Diogo Costa",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def create_data_frame() -> pd.DataFrame:
    #df = pd.read_csv('dataset.csv', sep=",", encoding="latin1", dtype=str)
    df_url = 'https://raw.githubusercontent.com/dioogo/ia003_data/refs/heads/main/dataset.csv'
    url_content = requests.get(df_url, verify=False).content
    df = pd.read_csv(io.StringIO(url_content.decode('utf-8')))
    df['Diabetes_binary'] = df['Diabetes_binary'].astype(float)
    df['BMI'] = df['BMI'].astype(float)
    df['Age'] = df['Age'].astype(float)
    df['Sex'] = df['Sex'].astype(float)
    df['PhysHlth'] = df['PhysHlth'].astype(float)
    df['DiffWalk'] = df['DiffWalk'].astype(float)
    df['GoodPhysHlth'] = 30 - df['PhysHlth']
    df['GenHlth'] = df['GenHlth'].astype(float)
    df['GoodHealth'] = df['GenHlth'].apply(lambda x: 1 if x < 5 else 0)
    
    #return df.sample(5000, random_state=50)
    return df

data_frame = create_data_frame()
data_frame_diabetes = data_frame[data_frame["Diabetes_binary"] == 1]
data_frame_no_diabetes = data_frame[data_frame["Diabetes_binary"] == 0]

data_frame_sample = data_frame.sample(1063, random_state=50)
data_frame_sample_diabetes = data_frame_sample[data_frame_sample["Diabetes_binary"] == 1]
data_frame_sample_no_diabetes = data_frame_sample[data_frame_sample["Diabetes_binary"] == 0]

total = len(data_frame)
total_diabetes = data_frame["Diabetes_binary"].sum()
total_no_diabetes = total - total_diabetes

st.title("IA003 — Trabalho Final - Diogo Costa")

if data_frame.empty:
    st.info("Nenhum dado corresponde aos filtros. Selecione ao menos uma categoria ou amplie o período.")
    st.stop()

st.header("Base de Dados")
st.markdown("#### Link:")
st.markdown("https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
st.markdown("#### Descrição")
st.markdown("A base de dados os dados uma pesquisa feita anualmente por telefone nos EUA sobre a saúde média dos entrevistados. Em torno de 400.000 americanos são entrevistados por ano desde 1984. Essa base representa um recorte da pesquisa feita em 2015 com dados referentes a Diabetes.")
st.markdown(f"#### O dataset possui {data_frame.shape[0]} instâncias e {data_frame.shape[1]} variáveis")
st.markdown("#### População de interesse e unidade amostral")
st.markdown("A população de interesse são os americanos entrevistados na pesquisa de saúde realizada por telefone. A unidade amostral é o indivíduo respondente da pesquisa.")
st.markdown("#### Variáveis selecionadas para as análises")
st.markdown(
    """
    - `Diabetes_binary`: Indicador de diabetes (0 = não, 1 = sim)
    - `BMI`: Índice de massa corporal
    - `PhysHlth`: Número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim
    - `DiffWalk`: Dificuldade para caminhar ou subir escadas (0 = não, 1 = sim)
    - `GenHlth`: Autoavaliação da saúde geral (1 = excelente, 2 = muito boa, 3 = boa, 4 = regular, 5 = péssima)
    - `GoodHealth`: Indicador de boa saúde geral (0 = não (GenHlth = 5), 1 = sim (GenHlth < 5))

    As diabetes, embora também tenha como causa genética, está fortemente associada a fatores de estilo de vida, como dieta e atividade física. Seus sintomas, quando presentes, afetam a qualidade de vida dos indivíduos.
    As variáveis escolhidas nesse trabalho nos permitem analisar a relação entre diabetes e fatores de saúde, como índice de massa corporal, saúde física, dificuldade para caminhar e percepção geral de saúde.
    """
)
st.divider()

st.header("Dimensionamento e seleção da amostra")
col1, col2, col3 = st.columns(3)
col1.metric("`Tamanho da população`", f"{total}")
col2.metric("`Grau de confiança`", "95%")
col3.metric("`Margem de erro`", "3%")
st.markdown("### Tamanho da amostra: 1.063")

st.markdown(
    f"""
    A escolha de 95% de grau de confiança se dá por ser um padrão amplamente utilizado em pesquisas estatísticas. Já a margem de erro de 3% se dá para obter uma estimativa mais precisa dado o tamanho grande da população.

    """
)

st.divider()
st.header("Painéis de visualização")

st.subheader("Painel 1 - População")
col1, col2, col3 = st.columns(3)
col1.metric("`Diabetes`", f"{total_diabetes:,.0f}")
col1.metric("`Diabetes (%)`", f"{total_diabetes/total*100:,.2f}%")
col2.metric("`Sem Diabetes`", f"{total_no_diabetes:,.0f}")
col2.metric("`Sem Diabetes (%)`", f"{total_no_diabetes/total*100:,.2f}%")
col3.metric("`Total`", f"{total:,.0f}")
col3.metric("`Total (%)`", f"{total/total*100:,.2f}%")

st.markdown("#### Distribuição por idade e sexo")

chart = plots.get_sex_age_count_plot(data_frame)
st.altair_chart(chart, **LARGURA)

st.markdown("#### Distribuição do IMC")
chart = plots.imc_plot(data_frame, data_frame_diabetes, data_frame_no_diabetes)
st.altair_chart(chart, **LARGURA)

imc_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Média IMC": [
        data_frame["BMI"].mean(),
        data_frame_no_diabetes["BMI"].mean(),
        data_frame_diabetes["BMI"].mean()
    ],
    "Mediana IMC": [
        data_frame["BMI"].median(),
        data_frame_no_diabetes["BMI"].median(),
        data_frame_diabetes["BMI"].median()
    ],
    "Desvio Padrão IMC": [
        data_frame["BMI"].std(),
        data_frame_no_diabetes["BMI"].std(),
        data_frame_diabetes["BMI"].std()
    ]
})

st.table(imc_description, alt="Descrição")

st.markdown("#### Distribuição de dias de Saúde Física Ruim")
chart = plots.physical_health_plot(data_frame, data_frame_diabetes, data_frame_no_diabetes)
st.altair_chart(chart, **LARGURA)

physical_health_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Média de dias de Saúde Física Ruim": [
        data_frame["PhysHlth"].mean(),
        data_frame_no_diabetes["PhysHlth"].mean(),
        data_frame_diabetes["PhysHlth"].mean()
    ],
    "Mediana de dias de Saúde Física Ruim": [
        data_frame["PhysHlth"].median(),
        data_frame_no_diabetes["PhysHlth"].median(),
        data_frame_diabetes["PhysHlth"].median()
    ],
    "Desvio Padrão de dias de Saúde Física Ruim": [
        data_frame["PhysHlth"].std(),
        data_frame_no_diabetes["PhysHlth"].std(),
        data_frame_diabetes["PhysHlth"].std()
    ]
})
st.table(physical_health_description, alt="Descrição")

st.markdown("#### Dificuldade para Caminhar ou Subir Escadas (em %)")
total_proportion_diff_walk = data_frame['DiffWalk'].value_counts(normalize=True)*100
no_diabetes_proportion_diff_walk = data_frame_no_diabetes['DiffWalk'].value_counts(normalize=True)*100
diabetes_proportion_diff_walk = data_frame_diabetes['DiffWalk'].value_counts(normalize=True)*100

chart = plots.diff_walk_plot(total_proportion_diff_walk, no_diabetes_proportion_diff_walk, diabetes_proportion_diff_walk)
st.altair_chart(chart, **LARGURA)

diff_walk_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Sem dificuldade (%)": [
        total_proportion_diff_walk.get(0, 0),
        no_diabetes_proportion_diff_walk.get(0, 0),
        diabetes_proportion_diff_walk.get(0, 0)
    ],
    "Com dificuldade (%)": [
        total_proportion_diff_walk.get(1, 0),
        no_diabetes_proportion_diff_walk.get(1, 0),
        diabetes_proportion_diff_walk.get(1, 0)
    ]
})
st.table(diff_walk_description, alt="Descrição")

st.markdown("#### Boa Saúde (em %)")
total_proportion_good_health = data_frame['GoodHealth'].value_counts(normalize=True)*100
no_diabetes_proportion_good_health = data_frame_no_diabetes['GoodHealth'].value_counts(normalize=True)*100
diabetes_proportion_good_health = data_frame_diabetes['GoodHealth'].value_counts(normalize=True)*100

chart = plots.good_health_plot(total_proportion_good_health, no_diabetes_proportion_good_health, diabetes_proportion_good_health)
st.altair_chart(chart, **LARGURA)

good_health_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Boa saúde (%)": [
        total_proportion_good_health.get(1, 0),
        no_diabetes_proportion_good_health.get(1, 0),
        diabetes_proportion_good_health.get(1, 0)
    ],
    "Saúde ruim (%)": [
        total_proportion_good_health.get(0, 0),
        no_diabetes_proportion_good_health.get(0, 0),
        diabetes_proportion_good_health.get(0, 0)
    ]
})
st.table(good_health_description, alt="Descrição")

st.divider()

st.subheader("Painel 2 - Amostra")

col1, col2, col3 = st.columns(3)
col1.metric("`Diabetes`", f"{len(data_frame_sample_diabetes):,.0f}")
col1.metric("`Diabetes (%)`", f"{len(data_frame_sample_diabetes)/len(data_frame_sample)*100:,.2f}%")
col2.metric("`Sem Diabetes`", f"{len(data_frame_sample_no_diabetes):,.0f}")
col2.metric("`Sem Diabetes (%)`", f"{len(data_frame_sample_no_diabetes)/len(data_frame_sample)*100:,.2f}%")
col3.metric("`Total`", f"{len(data_frame_sample):,.0f}")
col3.metric("`Total (%)`", f"{len(data_frame_sample)/len(data_frame_sample)*100:,.2f}%")


imc_population_and_sample_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Média IMC População": [
        data_frame["BMI"].mean(),
        data_frame_no_diabetes["BMI"].mean(),
        data_frame_diabetes["BMI"].mean()
    ],
    "Média IMC Amostra": [
        data_frame_sample["BMI"].mean(),
        data_frame_sample_no_diabetes["BMI"].mean(),
        data_frame_sample_diabetes["BMI"].mean()
    ],
    "Mediana IMC População": [
        data_frame["BMI"].median(),
        data_frame_no_diabetes["BMI"].median(),
        data_frame_diabetes["BMI"].median()
    ],
    "Mediana IMC Amostra": [
        data_frame_sample["BMI"].median(),
        data_frame_sample_no_diabetes["BMI"].median(),
        data_frame_sample_diabetes["BMI"].median()
    ],
    "Desvio Padrão Amostra": [
        data_frame_sample["BMI"].std(),
        data_frame_sample_no_diabetes["BMI"].std(),
        data_frame_sample_diabetes["BMI"].std()
    ],
    "Desvio Padrão População": [
        data_frame["BMI"].std(),
        data_frame_no_diabetes["BMI"].std(),
        data_frame_diabetes["BMI"].std()
    ],
    "Desvio Padrão População": [
        data_frame["BMI"].std(),
        data_frame_no_diabetes["BMI"].std(),
        data_frame_diabetes["BMI"].std()
    ],
    "Desvio Padrão Amostra": [
        data_frame_sample["BMI"].std(),
        data_frame_sample_no_diabetes["BMI"].std(),
        data_frame_sample_diabetes["BMI"].std()
    ],
})
st.markdown("#### Distribuição do IMC")
chart = plots.mean_std_population_vs_sample_plot(
    "BMI", "IMC Médio", "IMC Médio ± Desvio Padrão: População vs Amostra",
    data_frame, data_frame_diabetes, data_frame_no_diabetes,
    data_frame_sample, data_frame_sample_diabetes, data_frame_sample_no_diabetes,
)
st.altair_chart(chart, **LARGURA)

st.table(imc_population_and_sample_description, alt="Descrição")

st.markdown("#### Distribuição da Saúde Física Ruim")
physical_health_population_and_sample_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Média de Saúde Física Ruim População": [
        data_frame["PhysHlth"].mean(),
        data_frame_no_diabetes["PhysHlth"].mean(),
        data_frame_diabetes["PhysHlth"].mean()
    ],
    "Média de Saúde Física Ruim Amostra": [
        data_frame_sample["PhysHlth"].mean(),
        data_frame_sample_no_diabetes["PhysHlth"].mean(),
        data_frame_sample_diabetes["PhysHlth"].mean()
    ],
    "Mediana de Saúde Física Ruim População": [
        data_frame["PhysHlth"].median(),
        data_frame_no_diabetes["PhysHlth"].median(),
        data_frame_diabetes["PhysHlth"].median()
    ],
    "Mediana de Saúde Física Ruim Amostra": [
        data_frame_sample["PhysHlth"].median(),
        data_frame_sample_no_diabetes["PhysHlth"].median(),
        data_frame_sample_diabetes["PhysHlth"].median()
    ],
    "Desvio Padrão de Saúde Física Ruim População": [
        data_frame["PhysHlth"].std(),
        data_frame_no_diabetes["PhysHlth"].std(),
        data_frame_diabetes["PhysHlth"].std()
    ],
    "Desvio Padrão de Saúde Física Ruim Amostra": [
        data_frame_sample["PhysHlth"].std(),
        data_frame_sample_no_diabetes["PhysHlth"].std(),
        data_frame_sample_diabetes["PhysHlth"].std()
    ]
})

chart = plots.mean_std_population_vs_sample_plot(
    "PhysHlth", "Saúde Física Ruim Média", "Saúde Física Ruim ± Desvio Padrão: População vs Amostra",
    data_frame, data_frame_diabetes, data_frame_no_diabetes,
    data_frame_sample, data_frame_sample_diabetes, data_frame_sample_no_diabetes,
)
st.altair_chart(chart, **LARGURA)
st.table(physical_health_population_and_sample_description, alt="Descrição")

st.markdown("#### Dificuldade para caminhar ou subir escadas (%)")
total_proportion_diff_walk_sample = data_frame_sample['DiffWalk'].value_counts(normalize=True)*100
no_diabetes_proportion_diff_walk_sample = data_frame_sample_no_diabetes['DiffWalk'].value_counts(normalize=True)*100
diabetes_proportion_diff_walk_sample = data_frame_sample_diabetes['DiffWalk'].value_counts(normalize=True)*100

diff_walk_population_and_sample_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Sem dificuldade População (%)": [
        total_proportion_diff_walk.get(0, 0),
        no_diabetes_proportion_diff_walk.get(0, 0),
        diabetes_proportion_diff_walk.get(0, 0)
    ],
    "Sem dificuldade Amostra (%)": [
        total_proportion_diff_walk_sample.get(0, 0),
        no_diabetes_proportion_diff_walk_sample.get(0, 0),
        diabetes_proportion_diff_walk_sample.get(0, 0)
    ],
    "Com dificuldade População (%)": [
        total_proportion_diff_walk.get(1, 0),
        no_diabetes_proportion_diff_walk.get(1, 0),
        diabetes_proportion_diff_walk.get(1, 0)
    ],
    "Com dificuldade Amostra (%)": [
        total_proportion_diff_walk_sample.get(1, 0),
        no_diabetes_proportion_diff_walk_sample.get(1, 0),
        diabetes_proportion_diff_walk_sample.get(1, 0)
    ]
})
chart = plots.proportion_population_vs_sample_plot(
    "DiffWalk",
    {0: "Sem dificuldade", 1: "Com dificuldade"},
    "Dificuldade para Caminhar: População vs Amostra",
    data_frame, data_frame_diabetes, data_frame_no_diabetes,
    data_frame_sample, data_frame_sample_diabetes, data_frame_sample_no_diabetes,
)
st.altair_chart(chart, **LARGURA)
st.table(diff_walk_population_and_sample_description, alt="Descrição")

st.markdown("#### Boa Saúde (em %)")
total_proportion_good_health_sample = data_frame_sample['GoodHealth'].value_counts(normalize=True)*100
no_diabetes_proportion_good_health_sample = data_frame_sample_no_diabetes['GoodHealth'].value_counts(normalize=True)*100
diabetes_proportion_good_health_sample = data_frame_sample_diabetes['GoodHealth'].value_counts(normalize=True)*100

good_health_population_and_sample_description = pd.DataFrame({
    "População": ["Total", "Sem diabetes", "Com diabetes"],
    "Boa Saúde População (%)": [
        total_proportion_good_health.get(1, 0),
        no_diabetes_proportion_good_health.get(1, 0),
        diabetes_proportion_good_health.get(1, 0)
    ],
    "Boa Saúde Amostra (%)": [
        total_proportion_good_health_sample.get(1, 0),
        no_diabetes_proportion_good_health_sample.get(1, 0),
        diabetes_proportion_good_health_sample.get(1, 0)
    ],
    "Saúde ruim População (%)": [
        total_proportion_good_health.get(0, 0),
        no_diabetes_proportion_good_health.get(0, 0),
        diabetes_proportion_good_health.get(0, 0)
    ],
    "Saúde ruim Amostra (%)": [
        total_proportion_good_health_sample.get(0, 0),
        no_diabetes_proportion_good_health_sample.get(0, 0),
        diabetes_proportion_good_health_sample.get(0, 0)
    ]
})
chart = plots.proportion_population_vs_sample_plot(
    "GoodHealth",
    {0: "Saúde ruim", 1: "Boa Saúde"},
    "Boa Saúde: População vs Amostra",
    data_frame, data_frame_diabetes, data_frame_no_diabetes,
    data_frame_sample, data_frame_sample_diabetes, data_frame_sample_no_diabetes,
)
st.altair_chart(chart, **LARGURA)
st.table(good_health_population_and_sample_description, alt="Descrição")

st.divider()

st.header("Painel 3 — Inferência estatística e comparação dos resultados")

st.markdown("### Inferência para parâmetro de proporção")
st.markdown("#### Fórmulas utilizadas")

st.markdown("**Proporções amostrais:**")
st.latex(r"\hat{p}_0 = \frac{\text{casos}_{\text{diabéticos}}}{n_{\text{diabéticos}}} \qquad \hat{p}_1 = \frac{\text{casos}_{\text{não diabéticos}}}{n_{\text{não diabéticos}}}")

st.markdown("**Erro padrão da diferença (para o IC):**")
st.latex(r"SE = \sqrt{\frac{\hat{p}_0(1-\hat{p}_0)}{n_0} + \frac{\hat{p}_1(1-\hat{p}_1)}{n_1}}")

st.markdown("**Estatística Z do teste (proporção agrupada):**")
st.latex(r"Z = \frac{\hat{p}_0 - \hat{p}_1}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_0} + \frac{1}{n_1}\right)}} \qquad \hat{p} = \frac{\text{casos}_0 + \text{casos}_1}{n_0 + n_1}")

st.markdown("**Intervalo de confiança de 95% para a diferença:**")
st.latex(r"IC_{95\%} = (\hat{p}_0 - \hat{p}_1) \pm z^* \cdot SE \qquad z^* = \Phi^{-1}(0{,}975) \approx 1{,}96")


def calculate_proportion(field):
    n_diabetes = len(data_frame_sample_diabetes)
    n_no_diabetes = len(data_frame_sample_no_diabetes)
    count_field_diabetes = data_frame_sample_diabetes[field].sum()
    count_field_no_diabetes = data_frame_sample_no_diabetes[field].sum()

    p0 = count_field_diabetes / n_diabetes
    p1 = count_field_no_diabetes / n_no_diabetes

    count_field = np.array([
        count_field_diabetes,
        count_field_no_diabetes
    ])
    n_obs = np.array([
        n_diabetes,
        n_no_diabetes
    ])

    z_stat, p_value = proportions_ztest(count_field, n_obs, alternative='two-sided')
    se = np.sqrt(
        (p0) * (1 - p0) / n_diabetes +
        (p1) * (1 - p1) / n_no_diabetes
    )

    z_star = norm.ppf(0.975)
    diff = p0 - p1
    lower = diff - z_star * se
    upper = diff + z_star * se

    return z_stat, p_value, se * 100, lower * 100, upper * 100, p0 * 100, p1 * 100



n_diabetes = len(data_frame_sample_diabetes)
n_no_diabetes = len(data_frame_sample_no_diabetes)
#count_diff_walk_diabetes = data_frame_sample_diabetes['DiffWalk'].sum()
#count_diff_walk_no_diabetes = data_frame_sample_no_diabetes['DiffWalk'].sum()

#p0 = count_diff_walk_diabetes / n_diabetes
#p1 = count_diff_walk_no_diabetes / n_no_diabetes

#count_walk = np.array([
#    count_diff_walk_diabetes,
#    count_diff_walk_no_diabetes
#])
#n_obs = np.array([
#    n_diabetes,
#    n_no_diabetes
#])

#z_stat, p_value = proportions_ztest(count_walk, n_obs, alternative='two-sided')
#se = np.sqrt(
#    (p0) * (1 - p0) / n_diabetes +
#    (p1) * (1 - p1) / n_no_diabetes
#)

#z_star = norm.ppf(0.975)
#diff = p0 - p1
#lower = diff - z_star * se
#upper = diff + z_star * se


diff_walk_z_stat, diff_walk_p_value, diff_walk_se, diff_walk_lower, diff_walk_upper, diff_walk_p0, diff_walk_p1 = calculate_proportion('DiffWalk')

population_diff = diabetes_proportion_diff_walk.get(1, 0) - no_diabetes_proportion_diff_walk.get(1, 0)
print(f"Population difference: {population_diff}, amostra diferenca: {diff_walk_p0 - diff_walk_p1}")

st.markdown(f"""
    ##### Parâmetro selecionado: **Dificuldade para caminhar ou subir escadas**

    - Hipótese nula (H0): A proporção de indivíduos com dificuldade para caminhar ou subir escadas é a mesma entre diabéticos e não diabéticos.
    - Hipótese alternativa (H1): A proporção de indivíduos com dificuldade para caminhar ou subir escadas é diferente entre diabéticos e não diabéticos.

    - Nível de significância: 5% (α = 0,05)

    Analisando os dados da amostra, se percebe uma diferença entre a proporção de indivíduos com dificuldade para caminhar ou subir escadas entre diabéticos e não diabéticos.
    Onde **{no_diabetes_proportion_diff_walk_sample.get(1, 0):.2f}%** dos não diabéticos apresentam dificuldade para caminhar ou subir escadas, enquanto **{diabetes_proportion_diff_walk_sample.get(1, 0):.2f}%** dos diabéticos apresentam a mesma dificuldade.

    Agora, precisamos analisar estatisticamente se essa diferente é significante.
    Para isso, podemos realizar um teste de proporção (Z-test) para duas proporções independentes, comparando a proporção de dificuldade para caminhar ou subir escadas entre diabéticos e não diabéticos na amostra.

    Resultados:
    - Z test = **{diff_walk_z_stat:.4f}**
    - p-value = **{diff_walk_p_value:.15f}**
    - p0 - p1 = **{diff_walk_p0 - diff_walk_p1:.4f}**
    - Erro padrão = **{diff_walk_se:.4f}**
    - Conclusão: **{"Rejeitar H0" if diff_walk_p_value < 0.05 else "Não rejeitar H0"}**

    Podemos avalizar a conclusão analisando o intervalo de confiança para 95%:
    - Intervalo de confiança (95%): **[{diff_walk_lower:.4f}, {diff_walk_upper:.4f}]**
    - Ou seja, o intervalo de confiança **{"possui 0, o que indica que a diferença não é estatisticamente significativa" if diff_walk_lower <= 0 <= diff_walk_upper else "não possui 0, o que indica que a diferença é estatisticamente significativa"}**.

    Como a diferença é {"positiva" if diff_walk_p0 - diff_walk_p1 > 0 else "negativa"}, isso indica que a proporção de indivíduos com dificuldade para caminhar ou subir escadas é {"maior" if diff_walk_p0 - diff_walk_p1 > 0 else "menor"} entre diabéticos em comparação aos não diabéticos.

    ###### Comparação com a população
    Sendo nosso intervalo de confiança **[{diff_walk_lower:.4f}, {diff_walk_upper:.4f}]**, podemos compará-lo com a diferença observada na população, que é **{population_diff:.4f}**.
    - Conclusão: **{"A diferença populacional está dentro do intervalo de confiança, indicando consistência com a amostra" if diff_walk_lower <= population_diff <= diff_walk_upper else "A diferença populacional está fora do intervalo de confiança, indicando possível discrepância com a amostra"}**.

""")

#count_good_health_diabetes = data_frame_sample_diabetes['GoodHealth'].sum()
#count_good_health_no_diabetes = data_frame_sample_no_diabetes['GoodHealth'].sum()

#p0 = count_good_health_diabetes / n_diabetes
#p1 = count_good_health_no_diabetes / n_no_diabetes

#count_good_health = np.array([
#    count_good_health_diabetes,
#    count_good_health_no_diabetes
#])

#n_obs = np.array([
#    n_diabetes,
 #   n_no_diabetes
#])

#z_stat, p_value = proportions_ztest(count_good_health, n_obs, alternative='two-sided')
#se = np.sqrt(
#    (p0) * (1 - p0) / n_diabetes +
#    (p1) * (1 - p1) / n_no_diabetes
#)

#z_star = norm.ppf(0.975)
#diff = p0 - p1
#lower = diff - z_star * se
#upper = diff + z_star * se

good_health_z_stat, good_health_p_value, good_health_se, good_health_lower, good_health_upper, good_health_p0, good_health_p1 = calculate_proportion('GoodHealth')

population_diff = diabetes_proportion_good_health.get(1, 0) - no_diabetes_proportion_good_health.get(1, 0)

st.markdown(f"""
    ##### Parâmetro selecionado: **Boa saúde**

    - Hipótese nula (H0): A proporção de indivíduos com boa saúde é a mesma entre diabéticos e não diabéticos.
    - Hipótese alternativa (H1): A proporção de indivíduos com boa saúde é diferente entre diabéticos e não diabéticos.

    - Nível de significância: 5% (α = 0,05)

    Analisando os dados da amostra, se percebe uma diferença entre a proporção de indivíduos com boa saúde entre diabéticos e não diabéticos.
    Onde **{no_diabetes_proportion_good_health_sample.get(1, 0):.2f}%** dos não diabéticos apresentam boa saúde, enquanto **{diabetes_proportion_good_health_sample.get(1, 0):.2f}%** dos diabéticos apresentam a mesma condição.

    Agora, precisamos analisar estatisticamente se essa diferente é significante.
    Para isso, podemos realizar um teste de proporção (Z-test) para duas proporções independentes, comparando a proporção de boa saúde entre diabéticos e não diabéticos na amostra.

    Resultados:
    - Z test = **{good_health_z_stat:.4f}**
    - p-value = **{good_health_p_value:.7f}**
    - p0 - p1 = **{good_health_p0 - good_health_p1:.4f}**
    - Erro padrão = **{good_health_se:.4f}**
    - Conclusão: **{"Rejeitar H0" if good_health_p_value < 0.05 else "Não rejeitar H0"}**

    Podemos avalizar a conclusão analisando o intervalo de confiança para 95%:
    - Intervalo de confiança (95%): **[{good_health_lower:.4f}, {good_health_upper:.4f}]**
    - Ou seja, o intervalo de confiança **{"possui 0, o que indica que a diferença não é estatisticamente significativa" if good_health_lower <= 0 <= good_health_upper else "não possui 0, o que indica que a diferença é estatisticamente significativa"}**.

    Como a diferença é {"positiva" if good_health_p0 - good_health_p1 > 0 else "negativa"}, isso indica que a proporção de indivíduos com boa saúde é {"maior" if good_health_p0 - good_health_p1 > 0 else "menor"} entre diabéticos em comparação aos não diabéticos.

    ###### Comparação com a população
    Sendo nosso intervalo de confiança **[{good_health_lower:.4f}, {good_health_upper:.4f}]**, podemos compará-lo com a diferença observada na população, que é **{population_diff:.4f}**.
    - Conclusão: **{"A diferença populacional está dentro do intervalo de confiança, indicando consistência com a amostra" if good_health_lower <= population_diff <= good_health_upper else "A diferença populacional está fora do intervalo de confiança, indicando possível discrepância com a amostra"}**.

""")

st.divider()

st.markdown("### Inferência para parâmetro de média")
st.markdown("#### Fórmulas utilizadas")

st.markdown("**Erro padrão da diferença entre médias (variâncias separadas):**")
st.latex(r"SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}")

st.markdown("**Estatística Z do teste:**")
st.latex(r"Z = \frac{(\bar{x}_2 - \bar{x}_1) - \Delta_0}{SE} \qquad \Delta_0 = 0")

st.markdown("**Valor crítico para 95% de confiança:**")
st.latex(r"z_{\alpha/2} = \Phi^{-1}(0{,}975) \approx 1{,}96")

st.markdown("**Intervalo de confiança de 95% para a diferença de médias:**")
st.latex(r"IC_{95\%} = (\bar{x}_2 - \bar{x}_1) \pm z_{\alpha/2} \cdot SE")

def calculate_mean(field):
    field_no_diabetes_sample = data_frame_sample_no_diabetes[field]
    field_diabetes_sample = data_frame_sample_diabetes[field]

    dsw1 = ws.DescrStatsW(field_no_diabetes_sample)
    dsw2 = ws.DescrStatsW(field_diabetes_sample)
    cm = ws.CompareMeans(dsw1, dsw2)

    ht = cm.ztest_ind(alternative = 'two-sided', usevar = 'unequal', value = 0)

    # calcule e imprima o erro padrão, o escore Z e o valor-p para o teste de hipótese.
    se = cm.std_meandiff_separatevar
    testZ = ht[0]
    p_value = ht[1]

    z_critical = stats.norm.ppf(0.975)

    field_mean_no_diabetes = dsw1.mean
    field_mean_diabetes = dsw2.mean
    diff2 = field_mean_diabetes - field_mean_no_diabetes
    lower = diff2 - z_critical * se
    upper = diff2 + z_critical * se

    return testZ, se, p_value, lower, upper, field_mean_no_diabetes, field_mean_diabetes



imc_testZ, imc_se, imc_p_value, imc_lower, imc_upper, imc_mean_no_diabetes, imc_mean_diabetes = calculate_mean('BMI')




#imc_no_diabetes_sample = data_frame_sample_no_diabetes['BMI']
#imc_diabetes_sample = data_frame_sample_diabetes['BMI']

#dsw1 = ws.DescrStatsW(imc_no_diabetes_sample)
#dsw2 = ws.DescrStatsW(imc_diabetes_sample)
#cm = ws.CompareMeans(dsw1, dsw2)

#ht = cm.ztest_ind(alternative = 'two-sided', usevar = 'unequal', value = 0)

# calcule e imprima o erro padrão, o escore Z e o valor-p para o teste de hipótese.
# se = cm.std_meandiff_separatevar
#testZ = ht[0]
#p_value = ht[1]

#z_critical = stats.norm.ppf(0.975)

#mean_imc_no_diabetes = dsw1.mean
#mean_imc_diabetes = dsw2.mean
#diff2 = mean_imc_diabetes - mean_imc_no_diabetes
#ci_f_nf_low = diff2 - z_critical * se
#ci_f_nf_high = diff2 + z_critical * se




population_diff = imc_mean_diabetes - imc_mean_no_diabetes

st.markdown(f"""
    ##### Parâmetro selecionado: **IMC**

    - Hipótese nula (H0): A média do IMC é igual entre diabéticos e não diabéticos
    - Hipótese alternativa (H1): A média do IMC é diferente entre diabéticos e não diabéticos

    - Nível de significância: 5% (α = 0,05)

    Analisando os dados da amostra, identifica-se que a média do IMC de diabéticos ({imc_mean_diabetes:.4f}) é maior do que a média do IMC de não diabéticos ({imc_mean_no_diabetes:.4f}).

    Agora, precisamos analisar estatisticamente se essa diferente é significante.
    Para isso, podemos realizar um teste de média (Z-test) para duas médias independentes, comparando o IMC entre diabéticos e não diabéticos na amostra.

    Resultados:
    - Z test = **{imc_testZ:.4f}**
    - p-value = **{imc_p_value:.15f}**
    - Erro padrão = **{imc_se:.4f}**
    - Conclusão: **{"Rejeitar H0" if imc_p_value < 0.05 else "Não rejeitar H0"}**

    Podemos avalizar a conclusão analisando o intervalo de confiança para 95%:
    - Intervalo de confiança (95%): **[{imc_lower:.4f}, {imc_upper:.4f}]**
    - Ou seja, o intervalo de confiança **{"possui 0, o que indica que a diferença não é estatisticamente significativa" if imc_lower <= 0 <= imc_upper else "não possui 0, o que indica que a diferença é estatisticamente significativa"}**.

    Como a diferença é {"positiva" if imc_lower > 0 else "negativa"}, isso indica que a média do IMC de diabéticos é {"maior" if imc_lower > 0 else "menor"} entre diabéticos em comparação aos não diabéticos.

    ###### Comparação com a população
    Sendo nosso intervalo de confiança **[{imc_lower:.4f}, {imc_upper:.4f}]**, podemos compará-lo com a diferença observada na população, que é **{population_diff:.4f}**.
    - Conclusão: **{"A diferença populacional está dentro do intervalo de confiança, indicando consistência com a amostra" if imc_lower <= population_diff <= imc_upper else "A diferença populacional está fora do intervalo de confiança, indicando possível discrepância com a amostra"}**.
""")

#physical_health_no_diabetes_sample = data_frame_sample_no_diabetes['PhysHlth']
#physical_health_diabetes_sample = data_frame_sample_diabetes['PhysHlth']

#dsw1 = ws.DescrStatsW(physical_health_no_diabetes_sample)
#dsw2 = ws.DescrStatsW(physical_health_diabetes_sample)
#cm = ws.CompareMeans(dsw1, dsw2)

#ht = cm.ztest_ind(alternative = 'two-sided', usevar = 'unequal', value = 0)

# calcule e imprima o erro padrão, o escore Z e o valor-p para o teste de hipótese.
#se = cm.std_meandiff_separatevar
#testZ = ht[0]
#p_value = ht[1]

#z_critical = stats.norm.ppf(0.975)

#mean_physical_health_no_diabetes = dsw1.mean
#mean_physical_health_diabetes = dsw2.mean
#se = cm.std_meandiff_separatevar
#z_critical = stats.norm.ppf(0.975)

#diff1 = mean_physical_health_diabetes - mean_physical_health_no_diabetes
#ci_nf_f_low = diff1 - z_critical * se
#ci_nf_f_high = diff1 + z_critical * se

physical_health_testZ, physical_health_se, physical_health_p_value, physical_health_lower, physical_health_upper, physical_health_mean_no_diabetes, physical_health_mean_diabetes = calculate_mean('PhysHlth')

population_diff = physical_health_mean_diabetes - physical_health_mean_no_diabetes

st.markdown(f"""
    ##### Parâmetro selecionado: **Número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim**

    - Hipótese nula (H0): A média do número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim é igual entre diabéticos e não diabéticos
    - Hipótese alternativa (H1): A média do número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim é diferente entre diabéticos e não diabéticos

    - Nível de significância: 5% (α = 0,05)

    Analisando os dados da amostra, identifica-se que a média do número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim de diabéticos ({physical_health_mean_diabetes:.4f}) é maior do que a média do número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim de não diabéticos ({physical_health_mean_no_diabetes:.4f}).

    Agora, precisamos analisar estatisticamente se essa diferente é significante.
    Para isso, podemos realizar um teste de média (Z-test) para duas médias independentes, comparando o número de dias dos últimos 30 dias em que a pessoa apresentou saúde física ruim entre diabéticos e não diabéticos na amostra.

    Resultados:
    - Z test = **{physical_health_testZ:.4f}**
    - p-value = **{physical_health_p_value:.9f}**
    - Erro padrão = **{physical_health_se:.4f}**
    - Conclusão: **{"Rejeitar H0" if physical_health_p_value < 0.05 else "Não rejeitar H0"}**

    Podemos avalizar a conclusão analisando o intervalo de confiança para 95%:
    - Intervalo de confiança (95%): **[{physical_health_lower:.4f}, {physical_health_upper:.4f}]**
    - Ou seja, o intervalo de confiança **{"possui 0, o que indica que a diferença não é estatisticamente significativa" if physical_health_lower <= 0 <= physical_health_upper else "não possui 0, o que indica que a diferença é estatisticamente significativa"}**.

    Como a diferença é {"positiva" if physical_health_lower > 0 else "negativa"}, isso indica que a média de dias com saúde ruim nos últimos 30 dias de diabéticos é {"maior" if physical_health_lower > 0 else "menor"} entre diabéticos em comparação aos não diabéticos.

    ###### Comparação com a população
    Sendo nosso intervalo de confiança **[{physical_health_lower:.4f}, {physical_health_upper:.4f}]**, podemos compará-lo com a diferença observada na população, que é **{population_diff:.4f}**.
    - Conclusão: **{"A diferença populacional está dentro do intervalo de confiança, indicando consistência com a amostra" if physical_health_lower <= population_diff <= physical_health_upper else "A diferença populacional está fora do intervalo de confiança, indicando possível discrepância com a amostra"}**.

""")