# -*- coding: utf-8 -*-
"""
lab4.py
Лаба 4. Байесовские сети.
Анализ Титаника.
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Настройка отображения pandas для удобства чтения вывода в консоли
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

"""
# Байесовская сеть для Титаника

Суть лабы: строим граф зависимостей, чтобы понять, кто выжил, а кто нет.
Узлы графа — это признаки (пол, класс, порт), а стрелочки показывают, что на что влияет.
Наша цель — посмотреть, как пол, класс и место посадки влияют на шансы спастись.
"""

"""
Грузим данные и чистим их.
"""
file_path = "../lab02-classification/data/train.csv"
raw_df = pd.read_csv(file_path)

# Выбираем только признаки, участвующие в нашей байесовской сети
features = ["Survived", "Sex", "Pclass", "Embarked"]
df = raw_df[features].copy()

# Удаляем записи с пропущенным портом посадки
df.dropna(subset=["Embarked"], inplace=True)

# Явное приведение типов для корректной работы группировок
df["Survived"] = df["Survived"].astype(int)
df["Pclass"] = df["Pclass"].astype(int)
df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

print("Данные загружены. Первые 5 строк:")
print(df.head())


"""
## Как строим граф (наши гипотезы)

Связи между узлами рисуем сами, исходя из логики:

1. **Пол влияет на Выживание (Sex -> Survived)**:
   Потому что "сначала женщины и дети". Женщин спасали охотнее, значит пол важен.

2. **Класс влияет на Выживание (Pclass -> Survived)**:
   Богачи из 1-го класса жили ближе к палубе и шлюпкам, у них было больше шансов.

3. **Порт влияет на Класс (Embarked -> Pclass)**:
   В разных городах садились люди с разным достатком. Где-то больше богатых, где-то бедных мигрантов.
"""

bn = nx.DiGraph()
nodes = ["Embarked", "Pclass", "Sex", "Survived"]
edges = [("Embarked", "Pclass"), ("Pclass", "Survived"), ("Sex", "Survived")]
bn.add_nodes_from(nodes)
bn.add_edges_from(edges)

print("\nГраф построен:")
print(bn)

"""Рисуем граф, чтобы красиво было."""
plt.figure(figsize=(8, 5))
# Задаем фиксированное расположение узлов для наглядности
layout = {
    "Embarked": (-1, 1),  # Слева сверху
    "Pclass": (0, 1),  # По центру сверху
    "Sex": (1, 1),  # Справа сверху
    "Survived": (0, 0),  # Внизу (целевая переменная)
}

nx.draw(
    bn,
    pos=layout,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=10,
    arrowsize=20,
    arrowstyle="-|>",
)
plt.title("Структура Байесовской сети Titanic")
plt.show()


"""
## Считаем вероятности (CPT)

Для каждого узла нужно посчитать таблицу вероятностей.
Если у узла есть родители (например, Выживание зависит от Пола и Класса),
то мы считаем условную вероятность P(Ребенок | Родители).
Просто берем данные и считаем частоты.
"""

print("--- Простые вероятности (без условий) ---")

print("\nСколько кого по полу (Sex):")
# Считаем P(Sex)
prob_sex = df["Sex"].value_counts(normalize=True).reset_index()
prob_sex.columns = ["Sex", "probability"]
print(prob_sex)

print("\nСколько кого по портам (Embarked):")
# Считаем P(Embarked)
prob_embarked = df["Embarked"].value_counts(normalize=True).reset_index()
prob_embarked.columns = ["Embarked", "probability"]
print(prob_embarked)


print("\n--- Условные вероятности (кто от кого зависит) ---")

print("\nКак класс зависит от порта (P(Pclass | Embarked)):")
# Считаем P(Pclass | Embarked)
parents = ["Embarked"]
child = "Pclass"
group_fields = parents + [child]
# Считаем сколько раз встретилась комбинация
joint_freq = df.groupby(group_fields).size().reset_index(name="joint_count")
# Считаем сколько раз встретились просто Родители
parent_freq = df.groupby(parents).size().reset_index(name="parent_count")
# Делим одно на другое
merged = pd.merge(joint_freq, parent_freq, on=parents)
merged["probability"] = merged["joint_count"] / merged["parent_count"]
cpt_pclass = merged.drop(columns=["joint_count", "parent_count"])
# Сортируем и выводим
print(cpt_pclass.sort_values(["Embarked", "Pclass"]))

print("\nШансы выжить в зависимости от Пола и Класса (P(Survived | Sex, Pclass)):")
# Считаем P(Survived | Sex, Pclass)
parents = ["Sex", "Pclass"]
child = "Survived"
group_fields = parents + [child]
# Считаем сколько раз встретилась комбинация
joint_freq = df.groupby(group_fields).size().reset_index(name="joint_count")
# Считаем сколько раз встретились просто Родители
parent_freq = df.groupby(parents).size().reset_index(name="parent_count")
# Делим одно на другое
merged = pd.merge(joint_freq, parent_freq, on=parents)
merged["probability"] = merged["joint_count"] / merged["parent_count"]
cpt_survived = merged.drop(columns=["joint_count", "parent_count"])
# Сортируем и выводим
print(cpt_survived.sort_values(["Sex", "Pclass", "Survived"]))


"""
## Что получилось

Проверим, работает ли наша логика.
Посмотрим вероятности выживания для разных людей.
По идее, женщины из 1-го класса должны выживать чаще всех, а мужчины из 3-го — реже всех.
"""

print("\n=== Примерчики выживания ===")

# Сценарии для проверки
scenarios = [
    ("female", 1, "Женщина, 1 класс"),
    ("female", 3, "Женщина, 3 класс"),
    ("male", 1, "Мужчина, 1 класс"),
    ("male", 3, "Мужчина, 3 класс"),
]

for sex, pclass, desc in scenarios:
    # Фильтруем данные и считаем среднее (это и есть вероятность выжить)
    mask = (df["Sex"] == sex) & (df["Pclass"] == pclass)
    prob = df.loc[mask, "Survived"].mean()
    print(f"Шансы выжить ({desc}): {prob:.2%}")

print("\nКороче: гипотезы подтвердились. Пол и Класс реально решают.")
