import re

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
import unicodedata

def get_sex_age_count_plot(data_frame):
    age_range = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    age_range_js = "[" + ", ".join(f'"{x}"' for x in age_range) + "]"
    base = alt.Chart(data_frame).transform_calculate(
    gender=alt.expr.if_(alt.datum.Sex == 0, 'Female', 'Male'),
    age_label=f"{age_range_js}[datum.Age - 1]"   # Age 1→'18-24', 2→'25-29', …
).properties(width=250)
    color_scale = alt.Scale(domain=['Male', 'Female'],
                        range=['#1f77b4', '#e377c2'])
    left = base.transform_filter(
    alt.datum.gender == 'Female'
).encode(
    alt.Y('age_label:N').axis(None),
    alt.X('count():Q')
        .title('population')
        .sort('descending'),
    alt.Color('gender:N')
        .scale(color_scale)
        .legend(None)
).mark_bar().properties(title='Mulheres')

    middle = base.encode(
    alt.Y('age_label:N').axis(None),
    alt.Text('age_label:N'),
).mark_text().properties(width=20)

    right = base.transform_filter(
    alt.datum.gender == 'Male'
).encode(
    alt.Y('age_label:N').axis(None),
    alt.X('count():Q').title('population'),
    alt.Color('gender:N').scale(color_scale).legend(None)
).mark_bar().properties(title='Homens')
    return alt.hconcat(left, middle, right, spacing=5)

def imc_plot(data_frame, data_frame_diabetes, data_frame_no_diabetes):
    color_scale = alt.Scale(
        domain=['Total', 'Sem diabetes', 'Com diabetes'],
        range=['blue', 'green', 'red']
    )

    def mean_rule(df, color):
        return alt.Chart(df).mark_rule(color=color, strokeWidth=2, strokeDash=[6, 3]).encode(
            x=alt.X("mean(BMI):Q"),
            tooltip=alt.Tooltip("mean(BMI):Q", title="Média IMC", format=".2f")
        )

    a = alt.Chart(data_frame).transform_calculate(grupo='"Total"').mark_bar(opacity=0.5).encode(
        x=alt.X("BMI:Q", title="IMC",  bin=alt.Bin(step=3)),
        y=alt.Y("count()", title="Quantidade"),
        color=alt.Color("grupo:N", scale=color_scale, title="Grupo"),
    ).properties(width=650, height=400, title="População Total")

    b = alt.Chart(data_frame_no_diabetes).transform_calculate(grupo='"Sem diabetes"').mark_bar(opacity=0.3).encode(
        x=alt.X("BMI:Q", title="IMC",  bin=alt.Bin(step=3)),
        y=alt.Y("count()", title="Quantidade"),
        color=alt.Color("grupo:N", scale=color_scale, title="Grupo"),
    ).properties(width=650, height=400, title="População Sem Diabetes")

    c = alt.Chart(data_frame_diabetes).transform_calculate(grupo='"Com diabetes"').mark_bar(opacity=0.5).encode(
        x=alt.X("BMI:Q", title="IMC",  bin=alt.Bin(step=3)),
        y=alt.Y("count()", title="Quantidade"),
        color=alt.Color("grupo:N", scale=color_scale, title="Grupo"),
    ).properties(width=650, height=400, title="População Com Diabetes")

    return alt.hconcat(
        a + mean_rule(data_frame, 'blue'),
        b + mean_rule(data_frame_no_diabetes, 'green'),
        c + mean_rule(data_frame_diabetes, 'red'),
    )

def physical_health_plot(data_frame, data_frame_diabetes, data_frame_no_diabetes):
    color_scale = alt.Scale(
        domain=['Total', 'Sem diabetes', 'Com diabetes'],
        range=['blue', 'green', 'red']
    )

    def mean_rule(df, color):
        return alt.Chart(df).mark_rule(color=color, strokeWidth=2, strokeDash=[6, 3]).encode(
            x=alt.X("mean(PhysHlth):Q"),
            tooltip=alt.Tooltip("mean(PhysHlth):Q", title="Média de dias de Saúde Física Ruim", format=".2f")
        )

    a = alt.Chart(data_frame).transform_calculate(grupo='"Total"').mark_bar(opacity=0.5).encode(
        x=alt.X("PhysHlth:Q", title="Saúde Física Ruim",  bin=alt.Bin(step=3)),
        y=alt.Y("count()", title="Quantidade"),
        color=alt.Color("grupo:N", scale=color_scale, title="Grupo"),
    ).properties(width=650, height=400, title="População Total")

    b = alt.Chart(data_frame_no_diabetes).transform_calculate(grupo='"Sem diabetes"').mark_bar(opacity=0.3).encode(
        x=alt.X("PhysHlth:Q", title="Saúde Física Ruim",  bin=alt.Bin(step=3)),
        y=alt.Y("count()", title="Quantidade"),
        color=alt.Color("grupo:N", scale=color_scale, title="Grupo"),
    ).properties(width=650, height=400, title="População Sem Diabetes")

    c = alt.Chart(data_frame_diabetes).transform_calculate(grupo='"Com diabetes"').mark_bar(opacity=0.5).encode(
        x=alt.X("PhysHlth:Q", title="Saúde Física Ruim",  bin=alt.Bin(step=3)),
        y=alt.Y("count()", title="Quantidade"),
        color=alt.Color("grupo:N", scale=color_scale, title="Grupo"),
    ).properties(width=650, height=400, title="População Com Diabetes")

    return alt.hconcat(
        a + mean_rule(data_frame, 'blue'),
        b + mean_rule(data_frame_no_diabetes, 'green'),
        c + mean_rule(data_frame_diabetes, 'red'),
    )

def diff_walk_plot(total_proportion_diff_walk, no_diabetes_proportion_diff_walk, diabetes_proportion_diff_walk):
    diff_walk_df = pd.DataFrame({
        "Grupo": ["Total", "Sem diabetes", "Com diabetes"],
        "Sem dificuldade": [
            total_proportion_diff_walk.get(0, 0),
            no_diabetes_proportion_diff_walk.get(0, 0),
            diabetes_proportion_diff_walk.get(0, 0)
        ],
        "Com dificuldade": [
            total_proportion_diff_walk.get(1, 0),
            no_diabetes_proportion_diff_walk.get(1, 0),
            diabetes_proportion_diff_walk.get(1, 0)
        ]
    })

    return alt.Chart(diff_walk_df.melt(id_vars="Grupo", var_name="Dificuldade", value_name="Proporção")).mark_bar().encode(
        x=alt.X("Grupo:N", title="Grupo", sort=["Total", "Sem diabetes", "Com diabetes"]),
        y=alt.Y("Proporção:Q", title="Proporção"),
        color=alt.Color("Dificuldade:N", scale=alt.Scale(domain=["Sem dificuldade", "Com dificuldade"], range=["green", "red"]), title="Dificuldade")
    ).properties(width=650, height=400)

def good_health_plot(total_proportion_good_health, no_diabetes_proportion_good_health, diabetes_proportion_good_health):
    diff_walk_df = pd.DataFrame({
        "Grupo": ["Total", "Sem diabetes", "Com diabetes"],
        "Boa saúde": [
            total_proportion_good_health.get(1, 0),
            no_diabetes_proportion_good_health.get(1, 0),
            diabetes_proportion_good_health.get(1, 0)
        ],
        "Saúde ruim": [
            total_proportion_good_health.get(0, 0),
            no_diabetes_proportion_good_health.get(0, 0),
            diabetes_proportion_good_health.get(0, 0)
        ]
    })

    return alt.Chart(diff_walk_df.melt(id_vars="Grupo", var_name="Saúde", value_name="Proporção")).mark_bar().encode(
        x=alt.X("Grupo:N", title="Grupo", sort=["Total", "Sem diabetes", "Com diabetes"]),
        y=alt.Y("Proporção:Q", title="Proporção"),
        color=alt.Color("Saúde:N", scale=alt.Scale(domain=["Boa saúde", "Saúde ruim"], range=["green", "red"]), title="Saúde")
    ).properties(width=650, height=400)

def proportion_population_vs_sample_plot(column, category_labels, chart_title,
                                          data_frame, data_frame_diabetes, data_frame_no_diabetes,
                                          data_frame_sample, data_frame_sample_diabetes, data_frame_sample_no_diabetes):
    groups = [
        ("Total", data_frame, data_frame_sample),
        ("Sem diabetes", data_frame_no_diabetes, data_frame_sample_no_diabetes),
        ("Com diabetes", data_frame_diabetes, data_frame_sample_diabetes),
    ]
    rows = []
    for grupo, df_pop, df_samp in groups:
        for fonte, df in [("População", df_pop), ("Amostra", df_samp)]:
            props = df[column].value_counts(normalize=True) * 100
            for val, label in category_labels.items():
                rows.append({
                    "Grupo": grupo,
                    "Fonte": fonte,
                    "Categoria": label,
                    "Proporção (%)": props.get(val, 0),
                })

    chart_df = pd.DataFrame(rows)
    group_order = ["Total", "Sem diabetes", "Com diabetes"]
    category_order = list(category_labels.values())

    return alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Grupo:N", title="Grupo", sort=group_order),
        xOffset=alt.XOffset("Fonte:N"),
        y=alt.Y("Proporção (%):Q", title="Proporção (%)", stack=True),
        color=alt.Color("Categoria:N", sort=category_order, title="Categoria"),
        order=alt.Order("Categoria:N", sort="ascending"),
        tooltip=[
            alt.Tooltip("Grupo:N", title="Grupo"),
            alt.Tooltip("Fonte:N", title="Fonte"),
            alt.Tooltip("Categoria:N", title="Categoria"),
            alt.Tooltip("Proporção (%):Q", format=".2f", title="Proporção (%)"),
        ],
    ).properties(width=500, height=400, title=chart_title)


def mean_std_population_vs_sample_plot(column, y_title, chart_title,
                                        data_frame, data_frame_diabetes, data_frame_no_diabetes,
                                        data_frame_sample, data_frame_sample_diabetes, data_frame_sample_no_diabetes):
    groups = [
        ("Total", data_frame, data_frame_sample),
        ("Sem diabetes", data_frame_no_diabetes, data_frame_sample_no_diabetes),
        ("Com diabetes", data_frame_diabetes, data_frame_sample_diabetes),
    ]
    rows = []
    for grupo, df_pop, df_samp in groups:
        for fonte, df in [("População", df_pop), ("Amostra", df_samp)]:
            mean = df[column].mean()
            std = df[column].std()
            rows.append({
                "Grupo": grupo,
                "Fonte": fonte,
                "Média": mean,
                "Desvio Padrão": std,
                "lower": mean - std,
                "upper": mean + std,
            })

    chart_df = pd.DataFrame(rows)
    group_order = ["Total", "Sem diabetes", "Com diabetes"]

    color_scale = alt.Scale(domain=["População", "Amostra"], range=["#1f77b4", "#ff7f0e"])

    base = alt.Chart(chart_df)

    bars = base.mark_bar().encode(
        x=alt.X("Grupo:N", title="Grupo", sort=group_order),
        xOffset=alt.XOffset("Fonte:N"),
        y=alt.Y("Média:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("Fonte:N", scale=color_scale, title="Fonte"),
        tooltip=[
            alt.Tooltip("Grupo:N", title="Grupo"),
            alt.Tooltip("Fonte:N", title="Fonte"),
            alt.Tooltip("Média:Q", format=".2f", title=f"Média {y_title}"),
            alt.Tooltip("Desvio Padrão:Q", format=".2f", title="Desvio Padrão"),
        ],
    )

    error_rules = base.mark_rule(color="black", strokeWidth=2).encode(
        x=alt.X("Grupo:N", sort=group_order),
        xOffset=alt.XOffset("Fonte:N"),
        y=alt.Y("lower:Q", title=""),
        y2=alt.Y2("upper:Q"),
    )

    error_ticks_low = base.mark_tick(color="black", width=12).encode(
        x=alt.X("Grupo:N", sort=group_order),
        xOffset=alt.XOffset("Fonte:N"),
        y=alt.Y("lower:Q"),
    )

    error_ticks_high = base.mark_tick(color="black", width=12).encode(
        x=alt.X("Grupo:N", sort=group_order),
        xOffset=alt.XOffset("Fonte:N"),
        y=alt.Y("upper:Q"),
    )

    return (bars + error_rules + error_ticks_low + error_ticks_high).properties(
        width=500, height=400, title=chart_title
    )
